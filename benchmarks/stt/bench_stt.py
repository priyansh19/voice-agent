"""Benchmark: Kokoro (fallback TTS) synthesizes a test utterance, then Granite Speech 5.0 TurboCTC transcribes it."""
import time, sys, os
import numpy as np, soundfile as sf, torch
torch.set_num_threads(8)
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))

# --- Kokoro: make test audio (and time it) ---
from kokoro_onnx import Kokoro
t = time.perf_counter()
kk = Kokoro(f"{ROOT}/models/kokoro/kokoro-v1.0.onnx", f"{ROOT}/models/kokoro/voices-v1.0.bin")
print(f"[kokoro] load {time.perf_counter()-t:.2f}s")
text = "Hey, can you tell me what the weather is like in Milan tomorrow, and whether I should bring an umbrella?"
for i in range(2):
    t = time.perf_counter()
    wav, sr = kk.create(text, voice="am_adam", speed=1.0, lang="en-us")
    dt = time.perf_counter()-t
print(f"[kokoro] synth {dt*1000:.0f}ms for {len(wav)/sr:.2f}s audio (RTF {dt/(len(wav)/sr):.3f})")
sf.write(f"{ROOT}/samples/test_utterance_24k.wav", wav, sr)
# resample to 16k for STT
import torchaudio
wav16 = torchaudio.functional.resample(torch.from_numpy(wav).float(), sr, 16000).numpy()
sf.write(f"{ROOT}/samples/test_utterance_16k.wav", wav16, 16000)

# --- Granite Speech 5.0 TurboCTC ---
from transformers import AutoModelForCTC, AutoProcessor
mid = "ibm-granite/granite-speech-5.0-470m-turboctc"
t = time.perf_counter()
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForCTC.from_pretrained(mid, dtype=torch.float32).eval()
print(f"[granite-stt] load {time.perf_counter()-t:.2f}s  dtype={model.dtype}")
def transcribe(a):
    inputs = proc([a], sampling_rate=16000)
    with torch.inference_mode():
        out = model.generate(**inputs)
    return proc.batch_decode(out, skip_special_tokens=True)[0]
transcribe(wav16[:16000])  # warmup
for name, a in [("full", wav16), ("first1s", wav16[:16000]), ("first2s", wav16[:32000])]:
    t = time.perf_counter(); txt = transcribe(a); dt = time.perf_counter()-t
    print(f"[granite-stt] {name}: {dt*1000:.0f}ms for {len(a)/16000:.2f}s audio -> {txt!r}")
