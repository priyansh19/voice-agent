"""Transcribe TTS outputs with the STT to verify the bucket/text-padding trick keeps the words intact."""
import os, sys
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
os.chdir(ROOT)
import sys, os, glob, soundfile as sf, numpy as np
sys.path.insert(0, ROOT)
from agent import config as C
from agent.stt import GraniteSTT
from agent.audio_io import resample
cfg = C.load(); stt = GraniteSTT(cfg.stt); stt.warmup()
for p in sorted(glob.glob(C.abspath("samples/bk_*.wav"))):
    a, sr = sf.read(p, dtype="float32"); a16 = resample(a, sr, 16000)
    print(f"{os.path.basename(p):22s} {len(a)/sr:4.2f}s  rms={np.sqrt((a**2).mean()):.3f}  -> {stt.transcribe(a16)!r}")
