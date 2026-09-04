"""Silero VAD v6 endpointer with a *tentative* end-of-turn event.

Events (name, audio16k or None):
  speech_start   – speech confirmed (min_speech_ms of voiced frames)
  tentative_end  – first ~64 ms of silence after speech: the pipeline may start STT + LLM speculatively
  resumed        – speech came back before the firm end: cancel the speculative work
  end            – firm end of turn after `required_silence_ms` of silence
"""
import collections
import numpy as np
import torch
from silero_vad import load_silero_vad

# CTC transcripts have no punctuation; if the utterance ends on one of these, the user is probably mid-sentence.
CONTINUATION_WORDS = {"and", "but", "or", "so", "the", "a", "an", "to", "of", "in", "on", "at", "for", "with",
                      "that", "which", "if", "because", "then", "like", "um", "uh", "is", "are", "was", "i", "my"}


class Endpointer:
    def __init__(self, cfg, sample_rate=16000, frame_samples=512):
        self.cfg = cfg
        self.sr, self.n = sample_rate, frame_samples
        self.frame_ms = frame_samples / sample_rate * 1000
        self.model = load_silero_vad(onnx=True)
        self.thr = cfg.threshold
        self.neg_thr = max(0.05, cfg.threshold - 0.15)      # hysteresis (Silero v6 recommendation)
        self.pre_roll = collections.deque(maxlen=max(1, int(cfg.pre_roll_ms / self.frame_ms)))
        self.required_silence_ms = cfg.min_silence_ms
        self.noise_floor = 0.003            # running RMS of the room when nobody speaks
        self.min_rms_ratio = float(cfg.get("min_rms_ratio", 2.5))
        self.reset()

    def reset(self):
        self.state = "idle"
        self.buf = []
        self.onset_ms = 0.0
        self.silence_ms = 0.0
        self.speech_ms = 0.0
        self.required_silence_ms = self.cfg.min_silence_ms
        self.model.reset_states()

    def set_uncertain(self, transcript: str):
        """Called with the speculative transcript; extend the silence requirement if it looks unfinished."""
        last = transcript.strip().split(" ")[-1] if transcript.strip() else ""
        self.required_silence_ms = self.cfg.min_silence_uncertain_ms if last in CONTINUATION_WORDS else self.cfg.min_silence_ms

    def audio(self) -> np.ndarray:
        return np.concatenate(self.buf) if self.buf else np.zeros(0, dtype=np.float32)

    def process(self, frame: np.ndarray):
        """Feed one 32 ms frame; returns a list of (event, audio) tuples."""
        prob = float(self.model(torch.from_numpy(frame), self.sr).item())
        events = []
        if self.state == "idle":
            self.pre_roll.append(frame)
            rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
            if prob < 0.3:
                self.noise_floor = 0.97 * self.noise_floor + 0.03 * rms
            loud_enough = rms > self.noise_floor * self.min_rms_ratio
            if prob >= self.thr and loud_enough:
                self.onset_ms += self.frame_ms
                if self.onset_ms >= self.cfg.min_speech_ms:
                    self.buf = list(self.pre_roll)
                    self.state = "speech"; self.speech_ms = self.onset_ms; self.silence_ms = 0.0
                    events.append(("speech_start", None))
            else:
                self.onset_ms = 0.0
            return events
        self.buf.append(frame)
        self.speech_ms += self.frame_ms
        voiced = prob >= (self.thr if self.state == "trailing" else self.neg_thr)
        if self.state == "speech":
            if voiced:
                self.silence_ms = 0.0
            else:
                self.silence_ms += self.frame_ms
                if self.silence_ms >= 2 * self.frame_ms:
                    self.state = "trailing"
                    events.append(("tentative_end", self.audio()))
        elif self.state == "trailing":
            if voiced:
                self.state = "speech"; self.silence_ms = 0.0
                events.append(("resumed", None))
            else:
                self.silence_ms += self.frame_ms
                if self.silence_ms >= self.required_silence_ms:
                    events.append(("end", self.audio()))
                    self.reset()
                    return events
        if self.speech_ms / 1000 > self.cfg.max_utterance_s:
            events.append(("end", self.audio()))
            self.reset()
        return events
