"""Multilingual STT: Whisper large-v3-turbo (int4) through OpenVINO GenAI. ~0.85-1.1 s per utterance on the Arc iGPU,
auto language detection (99 languages incl. Hindi). Runs *alongside* the 130 ms English CTC model; the pipeline
switches to this transcript when the two disagree (non-English or English the CTC model missed)."""
import os, re, threading, time


class WhisperSTT:
    def __init__(self, cfg, log=print):
        import openvino_genai as og
        from huggingface_hub import snapshot_download
        self.log = log
        path = snapshot_download(cfg.model)
        t = time.perf_counter()
        self.pipe = og.WhisperPipeline(path, cfg.device, CACHE_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "whisper_ov_cache"))
        self.gc = self.pipe.get_generation_config()
        self.gc.max_new_tokens = 128; self.gc.task = "transcribe"; self.gc.return_timestamps = False
        self.lock = threading.Lock()
        log(f"[whisper] {cfg.model.split('/')[-1]} on {cfg.device} loaded in {time.perf_counter()-t:.1f}s")

    def warmup(self):
        import numpy as np
        self.transcribe(np.zeros(16000, dtype="float32"))

    def transcribe(self, audio16k) -> str:
        with self.lock:
            r = self.pipe.generate(audio16k.astype("float32").tolist(), self.gc)
        return r.texts[0].strip()


_norm = re.compile(r"[^\w\s]", re.UNICODE)


def same_utterance(a: str, b: str) -> bool:
    """True if two transcripts are the same words (ignoring case/punctuation), by word-set overlap."""
    wa, wb = set(_norm.sub("", a.lower()).split()), set(_norm.sub("", b.lower()).split())
    if not wa or not wb:
        return False
    j = len(wa & wb) / len(wa | wb)
    return j >= 0.5 or wa <= wb or wb <= wa


def is_non_latin(text: str) -> bool:
    return any(ord(c) > 0x24F for c in text)


def english_score(text: str) -> float:
    """Fraction of words that are common English words (wordfreq). Real English ~1.0; romanized Hindi ~0.3-0.6."""
    from wordfreq import zipf_frequency
    w = [x for x in text.lower().split() if x.isalpha()]
    if not w:
        return 0.0
    return sum(1 for x in w if zipf_frequency(x, "en") >= 3.0) / len(w)
