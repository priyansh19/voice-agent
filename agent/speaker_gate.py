"""Voice lock: only answer the enrolled speaker.
WeSpeaker CAM++ (Apache-2.0) speaker embeddings on onnxruntime; kaldi-style 80-dim fbank (torchaudio) with
per-utterance mean normalisation, exactly as WeSpeaker does at inference. Cosine similarity vs the enrolled voice.
"""
import os, time
import numpy as np


class SpeakerGate:
    def __init__(self, cfg, log=print):
        import onnxruntime as ort
        from .config import abspath
        self.cfg, self.log = cfg, log
        self.threshold = float(cfg.threshold)
        self.enabled = bool(cfg.enabled)
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(abspath(cfg.model), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.ref = None

    def _fbank(self, audio16k: np.ndarray) -> np.ndarray:
        import torch, torchaudio
        w = torch.from_numpy(audio16k.astype(np.float32)).unsqueeze(0) * 32768.0   # normalize_samples=0 -> int16 scale
        f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                                              sample_frequency=16000, window_type="hamming", use_energy=False)
        f = f - f.mean(dim=0, keepdim=True)
        return f.numpy()[None]

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        e = self.sess.run(None, {self.inp: self._fbank(audio16k)})[0][0].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)

    def enroll(self, wav_path):
        import soundfile as sf
        from .audio_io import resample
        a, sr = sf.read(wav_path, dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        a = resample(a, sr, 16000)
        win, hop = 16000 * 3, 16000          # average overlapping 3 s windows for a robust voiceprint
        fr = 320                              # drop silent windows (a quiet take is mostly silence)
        r = np.sqrt((a[: len(a) // fr * fr].reshape(-1, fr) ** 2).mean(axis=1))
        thr = max(0.003, 0.15 * float(r.max()) if len(r) else 0.003)
        def voiced_frac(seg):
            rr = np.sqrt((seg[: len(seg) // fr * fr].reshape(-1, fr) ** 2).mean(axis=1))
            return float((rr >= thr).mean()) if len(rr) else 0.0
        segs = [a[i:i + win] for i in range(0, max(1, len(a) - win + 1), hop)]
        good = [s for s in segs if voiced_frac(s) >= 0.4] or segs or [a]
        embs = [self.embed(s) for s in good]
        self.ref = np.mean(embs, axis=0); self.ref /= np.linalg.norm(self.ref) + 1e-8
        self.log(f"[gate] enrolled voice from {os.path.basename(wav_path)} ({len(a)/16000:.1f}s, {len(embs)} windows)")

    def similarity(self, audio16k: np.ndarray) -> float:
        if self.ref is None:
            return 1.0
        if len(audio16k) < 16000 // 2:
            audio16k = np.pad(audio16k, (0, 16000 // 2 - len(audio16k)))
        return float(self.embed(audio16k) @ self.ref)

    def accept(self, audio16k: np.ndarray):
        t = time.perf_counter()
        sim = self.similarity(audio16k)
        return (sim >= self.threshold) if self.enabled else True, sim, (time.perf_counter() - t) * 1000
