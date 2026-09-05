"""Multilingual STT: Whisper large-v3-turbo, auto language detection (99 languages incl. Hindi).
Runs *alongside* the fast English CTC model; the pipeline switches to this transcript when the two disagree.

Backends (config `stt.multilingual.backend`):
  openvino – OpenVINO GenAI on an Intel GPU (int4 model from the OpenVINO Hugging Face org), ~0.9 s per utterance.
  mlx      – mlx-whisper on Apple Silicon (Metal), model from mlx-community.
"""
import os, re, threading, time


class WhisperSTT:
    def __init__(self, cfg, log=print):
        self.log = log
        self.backend = cfg.get("backend", "openvino")
        self.lock = threading.Lock()
        t = time.perf_counter()
        if self.backend == "mlx":
            import mlx_whisper
            self.mlx = mlx_whisper
            self.model = cfg.model
            self._transcribe = self._mlx
            self.warm = False
        else:
            import openvino_genai as og
            from huggingface_hub import snapshot_download
            path = snapshot_download(cfg.model)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.pipe = og.WhisperPipeline(path, cfg.device, CACHE_DIR=os.path.join(root, "models", "whisper_ov_cache"))
            self.gc = self.pipe.get_generation_config()
            self.gc.max_new_tokens = 128; self.gc.task = "transcribe"; self.gc.return_timestamps = False
            self._transcribe = self._openvino
        log(f"[whisper] {cfg.model.split('/')[-1]} via {self.backend} ready in {time.perf_counter()-t:.1f}s")

    def _openvino(self, audio16k):
        r = self.pipe.generate(audio16k.astype("float32").tolist(), self.gc)
        return r.texts[0].strip()

    def _mlx(self, audio16k):
        r = self.mlx.transcribe(audio16k.astype("float32"), path_or_hf_repo=self.model, fp16=True,
                                condition_on_previous_text=False, temperature=0.0)
        return (r.get("text") or "").strip()

    def warmup(self):
        import numpy as np
        self.transcribe(np.zeros(16000, dtype="float32"))

    def transcribe(self, audio16k) -> str:
        with self.lock:
            return self._transcribe(audio16k)


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
    """Fraction of words that are common English words (wordfreq zipf >= 3). Romanized Hindi scores ~0.3-0.6."""
    from wordfreq import zipf_frequency
    words = [w for w in text.lower().split() if w.isalpha()]
    if not words:
        return 0.0
    return sum(1 for w in words if zipf_frequency(w, "en") >= 3.0) / len(words)
