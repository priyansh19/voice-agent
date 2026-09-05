"""TTS backends behind one interface:
   synth(text, on_done)  ->  on_done(audio48k float32, meta dict) called from a background thread, in submission order.
LuxWorkerTTS talks to tts_worker/server.py (cloned voice);  KokoroTTS runs in-process (generic fallback voice)."""
import json, os, re, subprocess, sys, threading, queue, time
import numpy as np
from .config import ROOT, abspath
from .audio_io import resample


class LuxWorkerTTS:
    def __init__(self, cfg, log=print):
        self.cfg, self.log = cfg, log
        venv = os.path.join(ROOT, "tts_worker", ".venv")
        py = os.path.join(venv, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv, "bin", "python")
        self.proc = subprocess.Popen([py, os.path.join(ROOT, "tts_worker", "server.py"), "--config", os.path.join(ROOT, "config.yaml")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=0,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTORCH_ENABLE_MPS_FALLBACK="1"))
        self.ready = threading.Event()
        self.pending = {}            # id -> callback
        self.fillers = {}            # text -> audio
        self.lock = threading.Lock()
        self._id = 0
        self.info = {}
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        out = self.proc.stdout
        while True:
            line = out.readline()
            if not line:
                self.log("[tts] worker exited"); self.ready.set(); return
            try:
                h = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            pcm = None
            if "samples" in h:
                n = int(h["samples"]) * 4
                buf = bytearray()
                while len(buf) < n:
                    chunk = out.read(n - len(buf))
                    if not chunk: break
                    buf += chunk
                pcm = np.frombuffer(bytes(buf), dtype=np.float32)
            if h.get("event") == "ready":
                self.info = h; self.ready.set()
            elif "error" in h and "id" not in h:
                self.log(f"[tts] worker error: {h['error']}"); self.ready.set()
            elif "filler" in h:
                self.fillers[h["filler"]] = pcm
            elif "id" in h:
                with self.lock:
                    cb = self.pending.pop(h["id"], None)
                if cb:
                    cb(pcm if pcm is not None else np.zeros(0, np.float32), h)

    def _send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8")); self.proc.stdin.flush()

    def wait_ready(self, timeout=900):
        self.ready.wait(timeout)
        return bool(self.info)

    def synth(self, text, on_done, steps=None):
        with self.lock:
            self._id += 1; rid = self._id; self.pending[rid] = on_done
        req = {"cmd": "synth", "id": rid, "text": text}
        if steps: req["steps"] = steps
        self._send(req)
        return rid

    def make_fillers(self, texts, timeout=120):
        self._send({"cmd": "fillers", "texts": list(texts)})
        t0 = time.perf_counter()
        while len(self.fillers) < len(texts) and time.perf_counter() - t0 < timeout:
            time.sleep(0.05)
        return self.fillers

    def close(self):
        try: self._send({"cmd": "quit"})
        except Exception: pass


class KokoroTTS:
    """Kokoro-82M (Apache-2.0), generic voice, in-process, sequential worker thread."""
    def __init__(self, cfg, log=print):
        from kokoro_onnx import Kokoro
        self.cfg, self.log = cfg, log
        self.k = Kokoro(abspath(cfg.kokoro.model), abspath(cfg.kokoro.voices))
        self.voice = cfg.kokoro.voice
        self.q = queue.Queue()
        self.fillers = {}
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text, cb, voice, lang = self.q.get()
            t = time.perf_counter()
            try:
                wav, sr = self.k.create(text, voice=voice or self.voice, speed=1.0, lang=lang or "en-us")
                wav = resample(np.asarray(wav, dtype=np.float32), sr, 48000)
            except Exception as e:
                self.log(f"[tts] kokoro failed: {e}"); wav = np.zeros(0, np.float32)
            cb(wav, {"total_ms": (time.perf_counter() - t) * 1000, "backend": "kokoro"})

    def wait_ready(self, timeout=None):
        done = threading.Event()
        self.synth("Warm up.", lambda w, m: done.set()); done.wait(60)
        return True

    def synth(self, text, on_done, steps=None, voice=None, lang=None):
        self.q.put((text, on_done, voice, lang))

    def make_fillers(self, texts, timeout=60):
        for t in texts:
            done = threading.Event()
            def cb(w, m, t=t, done=done): self.fillers[t] = w; done.set()
            self.synth(t, cb); done.wait(timeout)
        return self.fillers

    def close(self):
        pass


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class RoutedTTS:
    """Cloned voice (LuxTTS, English/Latin text) + Kokoro Hindi voice for Devanagari text."""
    def __init__(self, cfg, log=print):
        self.lux = LuxWorkerTTS(cfg, log)
        self.kok = KokoroTTS(cfg, log)
        self.hindi_voice = cfg.kokoro.get("hindi_voice", "hf_alpha")
        self.fillers = self.lux.fillers
    def wait_ready(self, timeout=900):
        return self.lux.wait_ready(timeout) and self.kok.wait_ready()
    def synth(self, text, on_done, steps=None):
        if _DEVANAGARI.search(text):
            return self.kok.synth(text, on_done, voice=self.hindi_voice, lang="hi")
        return self.lux.synth(text, on_done, steps)
    def make_fillers(self, texts, timeout=120):
        return self.lux.make_fillers(texts, timeout)
    def close(self):
        self.lux.close()


def make_tts(cfg, log=print):
    if cfg.backend == "lux":
        return RoutedTTS(cfg, log) if cfg.get("hindi_fallback", True) else LuxWorkerTTS(cfg, log)
    return KokoroTTS(cfg, log)
