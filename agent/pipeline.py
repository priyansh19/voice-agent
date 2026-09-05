"""Orchestrator: mic -> VAD endpointer -> (speculative) STT -> streaming LLM -> clause chunker -> TTS -> speaker.

Latency tricks (all measured in TurnMetrics):
  * speculative start: STT + LLM begin on the first silent frame, ~300 ms before the firm end of turn;
    if the user keeps talking the work is cancelled.
  * playback is gated on the firm end, so speculation never makes the agent interrupt the user.
  * adaptive endpoint: the speculative transcript decides whether the pause is a real end of turn.
  * first TTS chunk after only a few words; later chunks are full clauses; audio is generated ahead of playback.
  * optional cloned-voice filler ("Hmm.") played at the firm end while the real first sentence is produced.
"""
import os, random, threading, time, queue
import numpy as np
from .audio_io import Microphone, Speaker
from .vad import Endpointer
from .stt import GraniteSTT
from .llm import OllamaLLM
from .chunker import ClauseChunker
from .tts_client import make_tts
from .speaker_gate import SpeakerGate
from . import prefetch
from .metrics import TurnMetrics
from .config import abspath


class Turn:
    def __init__(self, tid, speculative):
        self.id = tid
        self.m = TurnMetrics(tid, speculative=speculative)
        self.cancel = threading.Event()
        self.committed = threading.Event()
        self.done_llm = threading.Event()
        self.audio_q = queue.Queue()      # (chunk_text, slot_queue) in order, ("__end__", None), or None = cancelled
        self.first_ready = threading.Event()
        self.first_audio_started = False
        self.transcript = ""
        self.response = ""


class VoiceAgent:
    def __init__(self, cfg, log=print, sink=None, components=None):
        """components: optional dict {stt, llm, tts, whisper, gate} of ready objects (tests inject fakes here)."""
        self.cfg, self.log = cfg, log
        c = components or {}
        self.ollama_proc = None
        if not c and cfg.llm.get("private_instance", False):
            from .ollama_launcher import ensure_ollama
            self.ollama_proc = ensure_ollama(cfg.llm.host, log)
        self.stt = c["stt"] if "stt" in c else GraniteSTT(cfg.stt, log)
        self.whisper = c.get("whisper")
        if not c and cfg.stt.get("multilingual", {}).get("enabled"):
            from .stt_whisper import WhisperSTT
            self.whisper = WhisperSTT(cfg.stt.multilingual, log)
        self.llm = c["llm"] if "llm" in c else OllamaLLM(cfg.llm, log)
        self.tts = c["tts"] if "tts" in c else make_tts(cfg.tts, log)
        self.speaker = Speaker(cfg.audio.output_sample_rate, cfg.audio.output_device, sink=sink)
        self.ep = Endpointer(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_samples)
        self.gate = c.get("gate") if c else (SpeakerGate(cfg.speaker_gate, log) if cfg.get("speaker_gate") else None)
        self.history = []
        self.turn = None
        self.turn_counter = 0
        self.lock = threading.Lock()
        self.fillers = {}
        self.tool_fillers = {}
        self.muted = False                       # UI mic gate
        self.pending_prefix = ""                 # transcript of a turn cancelled because the user kept talking
        self.file_feeding = False
        self.tts_steps = cfg.tts.steps
        self.state = "loading"
        self.on_event = lambda kind, data: None  # UI hook: ("state"|"transcript"|"delta"|"response"|"metrics"|"log", payload)

    def emit(self, kind, data):
        try:
            self.on_event(kind, data)
        except Exception:
            pass

    def set_state(self, s):
        if s != self.state:
            self.state = s; self.emit("state", s)

    def say(self, text):
        """Start a turn from text (skips STT). Used by the UI test box."""
        t = self._start_turn(None, speculative=False, t_tent=time.perf_counter(), text=text)
        t.m.speech_end = time.perf_counter(); self._commit(t)
        return t

    def candidate_path(self):
        p = abspath(self.cfg.tts.voice_ref)
        return os.path.join(os.path.dirname(p), "candidate.wav")

    def record_voice(self, seconds=20):
        """Record a candidate voice reference (48 kHz) with live level events; does NOT activate it yet."""
        import sounddevice as sd, soundfile as sf
        was = self.muted; self.muted = True
        chunks, t0, last = [], time.perf_counter(), [0.0]
        def cb(indata, frames, _time_info, status):
            x = indata[:, 0].copy(); chunks.append(x)
            el = time.perf_counter() - t0
            if el - last[0] >= 0.1:
                last[0] = el
                self.emit("rec_level", {"elapsed": el, "rms": float(np.sqrt((x * x).mean())), "peak": float(np.abs(x).max())})
        try:
            self.set_state("recording")
            self.log(f"[voice] recording {seconds:.0f}s...")
            with sd.InputStream(samplerate=48000, channels=1, dtype="float32", blocksize=2400,
                                device=self.cfg.audio.input_device, callback=cb):
                while time.perf_counter() - t0 < seconds:
                    time.sleep(0.05)
            a = np.concatenate(chunks)[: int(seconds * 48000)]
            peak = float(np.abs(a).max()) or 1e-6
            frame = 960                                            # 20 ms
            rms = np.sqrt((a[: len(a) // frame * frame].reshape(-1, frame) ** 2).mean(axis=1))
            thr = max(0.004, 0.03 * rms.max())
            silence_pct = float((rms < thr).mean() * 100)
            voiced = np.nonzero(rms >= thr)[0]
            lead = float(voiced[0] * frame / 48000) if len(voiced) else seconds
            clipped_pct = float((np.abs(a) > 0.985).mean() * 100)
            a = a[max(0, int((lead - 0.15) * 48000)):]              # trim lead-in silence (keep 150 ms)
            a = a / max(1e-6, float(np.abs(a).max())) * 0.9
            os.makedirs(os.path.dirname(self.candidate_path()), exist_ok=True); sf.write(self.candidate_path(), a, 48000)
            problems = []
            if peak < 0.15: problems.append("too quiet, move closer")
            if clipped_pct > 0.2: problems.append("clipping, lower the mic gain")
            if silence_pct > 45: problems.append("too much silence, keep talking")
            if lead > 2.5: problems.append("started late")
            stats = {"duration": len(a) / 48000, "peak": peak, "clipped_pct": clipped_pct, "silence_pct": silence_pct,
                     "lead_silence_s": lead, "ok": not problems, "verdict": "good take" if not problems else "; ".join(problems)}
            self.log(f"[voice] candidate saved: {stats}")
            self.emit("rec_done", stats)
        finally:
            self.muted = was; self.set_state("idle")

    def use_recording(self):
        """Promote the candidate to the active voice and reload clone + voice lock."""
        import shutil
        src, dst = self.candidate_path(), abspath(self.cfg.tts.voice_ref)
        if not os.path.exists(src):
            self.log("[voice] no candidate recording"); return
        shutil.copyfile(src, dst); self.log(f"[voice] {dst} updated")
        self.reload_voice()
        self.emit("voice_reloaded", {"path": dst})

    def test_voice(self, text="Hi, this is my cloned voice. If it sounds like me, we're good to go."):
        done = threading.Event(); out = {}
        def cb(w, meta): out["w"] = w; done.set()
        self.tts.synth(text, cb, steps=self.tts_steps); done.wait(30)
        if out.get("w") is not None and len(out["w"]):
            self.speaker.play(out["w"])

    def reload_voice(self):
        self.set_state("loading")
        self.log("[voice] re-encoding the cloned voice (TTS worker restart)...")
        try:
            try: self.tts.close()
            except Exception: pass
            self.tts = make_tts(self.cfg.tts, self.log)
            if not self.tts.wait_ready():
                raise RuntimeError("TTS worker failed to restart")
            if self.cfg.tts.fillers.enabled:
                texts = list(self.cfg.tts.fillers.texts) + list(self.cfg.tts.fillers.get("tool_texts", []))
                made = self.tts.make_fillers(texts)
                self.fillers = {k: v for k, v in made.items() if k in self.cfg.tts.fillers.texts}
                self.tool_fillers = {k: v for k, v in made.items() if k in self.cfg.tts.fillers.get("tool_texts", [])}
            if self.gate:
                self.gate.enroll(abspath(self.cfg.tts.voice_ref))
            self.log("[voice] new voice ready")
        except Exception as e:
            self.log(f"[voice] reload failed: {e!r}")
        finally:
            self.set_state("idle")

    def set_input_device(self, idx):
        """Switch the microphone (restarts the capture stream)."""
        self.cfg["audio"]["input_device"] = idx
        mic = getattr(self, "mic", None)
        if mic:
            mic.q.put(None)                 # ends the current feed loop; run_mic reopens with the new device
        self.log(f"[audio] input device -> {idx}")

    def run_wav(self, path):
        """Feed a wav file through VAD/STT as if spoken (offline test)."""
        import soundfile as sf
        from .audio_io import resample
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        a = np.concatenate([resample(a, sr, 16000), np.zeros(16000, np.float32)])
        n = self.cfg.audio.frame_samples
        def paced():                      # real-time pacing so speculative timing is realistic
            t0 = time.perf_counter()
            for i, f in enumerate(a[: len(a) // n * n].reshape(-1, n)):
                while time.perf_counter() - t0 < i * n / 16000: time.sleep(0.002)
                yield f
        self.file_feeding = True
        try:
            self.ep.reset()
            self.feed(paced(), ignore_mute=True)
        finally:
            self.file_feeding = False

    # ------------------------------------------------------------ lifecycle
    def warmup(self):
        self.log("[agent] warming up...")
        from . import tools
        tools.prewarm()
        self.stt.warmup()
        if self.whisper: self.whisper.warmup()
        self.llm.warmup()
        ok = self.tts.wait_ready()
        if not ok:
            raise RuntimeError("TTS worker failed to start (see messages above)")
        if self.cfg.tts.fillers.enabled:
            texts = list(self.cfg.tts.fillers.texts) + list(self.cfg.tts.fillers.get("tool_texts", []))
            made = self.tts.make_fillers(texts)
            self.fillers = {k: v for k, v in made.items() if k in self.cfg.tts.fillers.texts}
            self.tool_fillers = {k: v for k, v in made.items() if k in self.cfg.tts.fillers.get("tool_texts", [])}
            self.log(f"[agent] fillers ready: {list(made)}")
        if self.gate and os.path.exists(abspath(self.cfg.tts.voice_ref)):
            self.gate.enroll(abspath(self.cfg.tts.voice_ref))
        self.speaker.start()
        self.log("[agent] ready — speak.")
        self.set_state("idle")

    def close(self):
        self.tts.close(); self.speaker.close()
        if self.ollama_proc:
            try: self.ollama_proc.terminate()
            except Exception: pass

    # ------------------------------------------------------------ audio in
    def run_mic(self):
        while True:
            try:
                mic = Microphone(self.cfg.audio.sample_rate, self.cfg.audio.frame_samples, self.cfg.audio.input_device)
                mic.start()
            except Exception as e:
                self.log(f"[audio] cannot open input device {self.cfg.audio.input_device!r}: {e}; trying default")
                self.cfg["audio"]["input_device"] = None
                try:
                    mic = Microphone(self.cfg.audio.sample_rate, self.cfg.audio.frame_samples, None); mic.start()
                except Exception as e2:
                    self.log(f"[audio] no local microphone ({e2.__class__.__name__}); browser microphone mode only")
                    self.mic = None
                    return
            self.mic = mic
            try:
                self.feed(mic.frames())     # returns when set_input_device() posts a None frame
            finally:
                mic.stop()

    def attach_browser_audio(self, send_pcm, send_stop):
        """Route audio I/O through a browser: mic frames arrive via feed(source='browser'), playback goes to send_pcm."""
        self.browser_audio = True
        self.speaker.remote, self.speaker.remote_stop = send_pcm, send_stop
        self.ep.reset()
        self.log("[audio] browser microphone + playback attached (local devices paused)")

    def detach_browser_audio(self):
        self.browser_audio = False
        self.speaker.remote = self.speaker.remote_stop = None
        self.log("[audio] browser audio detached; local devices active")

    def feed(self, frames, ignore_mute=False, source="mic"):
        """Consume 32 ms float32 frames (from the mic, a browser, or a file)."""
        for frame in frames:
            if frame is None:
                return
            if source == "mic" and getattr(self, "browser_audio", False):
                continue                             # a browser owns the audio right now
            if self.muted and not ignore_mute:
                if self.ep.state != "idle" and not self.file_feeding: self.ep.reset()
                continue
            if self.file_feeding and not ignore_mute:
                continue                             # a file replay owns the endpointer right now
            if not self.cfg.vad.barge_in and self.speaker.is_busy():
                self.ep.reset(); continue            # laptop mic hears the speaker: ignore it while we talk
            for ev, audio in self.ep.process(frame):
                self._on_event(ev, audio)

    def _on_event(self, ev, audio):
        now = time.perf_counter()
        with self.lock:
            t = self.turn
        if ev == "speech_start":
            self.set_state("listening")
            if t and not t.cancel.is_set():
                if not t.first_audio_started:      # answer not audible yet: the user is continuing; merge into the next turn
                    if t.committed.is_set():
                        if t.transcript:
                            self.pending_prefix = t.transcript      # already contains any earlier prefix
                        else:
                            t.merge = True         # STT still running: _process will hand its text over
                    self._cancel(t)
                elif self.cfg.vad.barge_in:
                    self._cancel(t); self.speaker.stop()
        elif ev == "tentative_end":
            if self.cfg.vad.speculative and (t is None or t.cancel.is_set() or t.committed.is_set()):
                self._start_turn(audio, speculative=True, t_tent=now)
        elif ev == "resumed":
            if t and not t.committed.is_set():
                self._cancel(t)
        elif ev == "end":
            with self.lock:
                t = self.turn                       # may have been replaced by a multilingual override
            if t and t.m.speculative and not t.cancel.is_set() and not t.committed.is_set():
                t.m.speech_end = now; self._commit(t)
            else:
                nt = self._start_turn(audio, speculative=False, t_tent=now)
                nt.m.speech_end = now; self._commit(nt)

    # ------------------------------------------------------------ turns
    def _start_turn(self, audio, speculative, t_tent, text=None):
        with self.lock:
            self.turn_counter += 1
            t = Turn(self.turn_counter, speculative); t.m.tentative_end = t_tent
            self.turn = t
        threading.Thread(target=self._safe, args=(self._process, t, audio, text), daemon=True).start()
        threading.Thread(target=self._safe, args=(self._player, t), daemon=True).start()
        return t

    def _safe(self, fn, t, *args):
        try:
            fn(t, *args)
        except Exception as e:
            import traceback
            self.log(f"[agent] turn {t.id} failed in {fn.__name__}: {e!r}\n{traceback.format_exc(limit=3)}")
            self._cancel(t); self.set_state("idle")

    def _cancel(self, t):
        t.cancel.set(); t.m.cancelled = True
        t.audio_q.put(None)
        c = getattr(t, "llm_client", None)
        if c is not None:
            self.llm.abort(c)                  # free Ollama immediately for the next turn

    def _commit(self, t):
        t.committed.set()

    def _process(self, t: Turn, audio, text=None):
        m = t.m
        planned = None
        if text is None:
            if self.gate and self.gate.enabled and audio is not None:
                ok, sim, ms = self.gate.accept(audio)
                self.emit("gate", {"turn": t.id, "sim": round(sim, 3), "ok": ok, "ms": round(ms)})
                if not ok:
                    self.log(f"[gate] ignored: not your voice (similarity {sim:.2f} < {self.gate.threshold:.2f}, {ms:.0f}ms)")
                    self._cancel(t); self.set_state("idle"); return
                self.log(f"[gate] you (similarity {sim:.2f}, {ms:.0f}ms)")
            text = self.stt.transcribe(audio)
            self.log(f"[stt] turn {t.id}: {len(audio)/16000:.2f}s audio -> {text!r}")
            if self.whisper:
                from .stt_whisper import english_score
                score = english_score(text)
                if len(text.split()) >= 3 and score < self.cfg.stt.multilingual.get("english_threshold", 0.7):
                    # does not look like English: don't waste the GPU on it, wait for the multilingual model
                    t0 = time.perf_counter()
                    try:
                        wtext = self.whisper.transcribe(audio)
                    except Exception as e:
                        wtext = ""; self.log(f"[whisper] failed: {e!r}")
                    self.log(f"[whisper] used directly (english-likeness {score:.2f}, {(time.perf_counter()-t0)*1000:.0f}ms): {wtext!r}")
                    if wtext: text = wtext
                elif score < self.cfg.stt.multilingual.get("skip_check_above", 0.9) or len(text.split()) < 3:
                    # borderline: proceed now, verify in parallel (clear English skips this to keep the GPU free)
                    threading.Thread(target=self._whisper_check, args=(t, audio), daemon=True).start()
            if self.pending_prefix:
                text = (self.pending_prefix + " " + text).strip()
        m.stt_done = time.perf_counter(); m.transcript = text; t.transcript = text
        if t.cancel.is_set():
            if getattr(t, "merge", False) and text:
                self.pending_prefix = text                  # merged text (prefix + this segment)
            return
        if len(text.split()) == 0:
            self._cancel(t); return
        self.pending_prefix = ""
        planned = prefetch.plan(text, self.cfg.llm)          # e.g. weather: start fetching before the LLM runs
        if planned: self.log(f"[prefetch] started: {planned[0]}")
        if not t.committed.is_set():
            self.ep.set_uncertain(text)
        self.log(f"[you] {text}")
        self.emit("transcript", {"turn": t.id, "text": text, "speculative": t.m.speculative})
        self.set_state("thinking")
        chunker = ClauseChunker(self.cfg.chunker.first_chunk_min_words, self.cfg.chunker.min_words, self.cfg.chunker.max_chars,
                                self.cfg.chunker.get("first_chunk_max_words", 0))
        first = True
        def on_meta(part):
            m.llm_prompt_tokens, m.llm_eval_tokens = part.prompt_eval_count or 0, part.eval_count or 0
            m.llm_prompt_ms, m.llm_eval_ms = (part.prompt_eval_duration or 0) / 1e6, (part.eval_duration or 0) / 1e6
        def on_client(c): t.llm_client = c
        def on_tool(name):
            self.log(f"[agent] calling tool {name}...")
            self.emit("tool", {"turn": t.id, "name": name})
            if self.tool_fillers and not t.first_ready.is_set():        # say "Let me check that." while the tool runs
                slot = queue.Queue(maxsize=1); slot.put((random.choice(list(self.tool_fillers.values())), {"total_ms": 0}))
                t.audio_q.put(("(checking)", slot)); t.first_ready.set()
        lang = self._language(text)
        llm_text = f"[user spoke {lang}; reply in {lang}] " + text + prefetch.collect(planned, self.cfg.llm.get("prefetch_wait_s", 1.5), self.log)
        for delta in self.llm.stream(self.history, llm_text, t.cancel, on_meta, on_client, on_tool):
            if first:
                m.llm_first_token = time.perf_counter(); first = False
            t.response += delta
            self.emit("delta", {"turn": t.id, "text": delta})
            for chunk in chunker.feed(delta):
                self._tts(t, chunk)
        if t.cancel.is_set():
            return
        for chunk in chunker.flush():
            self._tts(t, chunk)
        m.llm_done = time.perf_counter(); m.response = t.response
        t.done_llm.set()
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": t.response.strip()}]
        self.history = self.history[-2 * self.cfg.llm.max_history_turns:]
        self.log(f"[agent] {t.response.strip()}")
        self.emit("response", {"turn": t.id, "text": t.response.strip()})
        t.audio_q.put(("__end__", None))            # after the response event, so listeners see it before audio_done

    @staticmethod
    def _language(text):
        from .stt_whisper import english_score
        if any("ऀ" <= c <= "ॿ" for c in text):
            return "Hindi (Devanagari script)"
        if any(ord(c) > 0x24F for c in text):
            return "the same language as the user, in the same script"
        return "English" if english_score(text) >= 0.7 else "Hinglish (Hindi in Latin letters)"

    def _whisper_check(self, t: Turn, audio):
        from .stt_whisper import same_utterance, is_non_latin
        t0 = time.perf_counter()
        try:
            wtext = self.whisper.transcribe(audio)
        except Exception as e:
            self.log(f"[whisper] failed: {e!r}"); return
        ms = (time.perf_counter() - t0) * 1000
        if t.cancel.is_set() or not wtext:
            return
        if not is_non_latin(wtext) and same_utterance(wtext, t.transcript):
            self.log(f"[whisper] agrees ({ms:.0f}ms): {wtext!r}"); return
        if t.first_audio_started:
            self.log(f"[whisper] too late to switch ({ms:.0f}ms): {wtext!r}"); return
        self.log(f"[whisper] override ({ms:.0f}ms): {wtext!r}")
        text = (self.pending_prefix + " " + wtext).strip() if self.pending_prefix else wtext
        nt = self._start_turn(None, speculative=t.m.speculative, t_tent=t.m.tentative_end, text=text)
        nt.m.speech_end = t.m.speech_end
        if t.committed.is_set():
            self._commit(nt)
        self._cancel(t)
        self.emit("transcript", {"turn": nt.id, "text": text, "speculative": False, "multilingual": True})

    def _tts(self, t: Turn, chunk):
        if not t.m.first_chunk_text:
            t.m.first_chunk_text = time.perf_counter()
        slot = queue.Queue(maxsize=1)
        t.audio_q.put((chunk, slot))               # reserve the ordered slot now, fill it when TTS finishes
        def done(wav, meta):
            slot.put((wav, meta)); t.first_ready.set()
        first = not t.first_ready.is_set() and t.audio_q.qsize() <= 1
        steps = min(self.tts_steps, self.cfg.tts.get("first_chunk_steps", self.tts_steps)) if first else self.tts_steps
        self.tts.synth(chunk, done, steps=steps)

    def _player(self, t: Turn):
        m = t.m
        t.committed.wait()
        if t.cancel.is_set():
            return
        # filler: if nothing is ready to play shortly after the firm end, say something in the cloned voice right away
        if self.fillers and not t.first_ready.wait(self.cfg.tts.fillers.after_ms / 1000) and not t.cancel.is_set():
            filler = random.choice(list(self.fillers))
            def fstart(): m.filler_played = time.perf_counter()
            self.speaker.play(self.fillers[filler], on_start=fstart)
        while True:
            item = t.audio_q.get()
            if item is None or t.cancel.is_set():
                return
            chunk, slot = item
            if chunk == "__end__":
                break
            got = None
            while got is None:
                try: got = slot.get(timeout=0.1)
                except queue.Empty:
                    if t.cancel.is_set(): return
            wav, meta = got
            m.tts_chunks.append(f"{meta.get('total_ms', 0):.0f}ms/{len(wav)/48000:.1f}s '{chunk[:30]}'")
            if not m.first_audio_ready and chunk != "(checking)":
                m.first_audio_ready = time.perf_counter()
            if len(wav) == 0:
                continue
            if chunk == "(checking)":            # tool filler: audible, but not the "answer"
                self.speaker.play(wav); continue
            def start(first=not t.first_audio_started):
                if first: m.first_audio_played = time.perf_counter()
                self.set_state("speaking")
            t.first_audio_started = True
            self.speaker.play(wav, on_start=start)
        self.speaker.wait_idle()
        done = time.perf_counter()
        m.audio_done = done
        m.report()
        self.emit("metrics", {"turn": t.id, **m.summary(), "rows": m.rows(), "transcript": m.transcript, "response": m.response})
        self.set_state("idle")
        t.finished = True                           # set last: everything about this turn has been published
