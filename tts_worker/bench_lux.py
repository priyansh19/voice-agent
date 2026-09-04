"""Benchmark LuxTTS voice cloning on CPU: prompt encode + per-sentence synth latency."""
import time, os, sys
import numpy as np, soundfile as sf, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ref = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/samples/test_utterance_24k.wav"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
from zipvoice.luxvoice import LuxTTS
t = time.perf_counter()
tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=threads)
print(f"[lux] load {time.perf_counter()-t:.1f}s threads={threads}")
t = time.perf_counter()
enc = tts.encode_prompt(ref, duration=5, rms=0.01)
print(f"[lux] encode_prompt {time.perf_counter()-t*1000 if False else (time.perf_counter()-t)*1000:.0f}ms")
sents = ["Sure, I can help with that.",
         "Tomorrow in Milan it should be mostly cloudy with a chance of light rain in the afternoon.",
         "Yes, bring an umbrella just in case."]
tts.generate_speech("This is a slightly longer warm up sentence for the model.", enc, num_steps=4)
import traceback
for short in ["Sure.", "Okay, sure.", "Yes, bring an umbrella.", "Hmm."]:
    try:
        t = time.perf_counter(); w = tts.generate_speech(short, enc, num_steps=4); dt=time.perf_counter()-t
        print(f"[lux] short ok {short!r}: {dt*1000:.0f}ms {w.numel()/48000:.2f}s")
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        frames = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in tb if 'zipvoice' in f.filename or 'linacodec' in f.filename]
        print(f"[lux] short FAIL {short!r}: {type(e).__name__}: {str(e)[:80]} @ {frames[-3:]}")
for steps in (2, 4):
    for s in sents:
        t = time.perf_counter()
        wav = tts.generate_speech(s, enc, num_steps=steps)
        dt = time.perf_counter()-t
        wav = wav.numpy().squeeze()
        print(f"[lux] steps={steps} {dt*1000:.0f}ms for {len(wav)/48000:.2f}s audio (RTF {dt/(len(wav)/48000):.2f}) :: {s[:40]}")
sf.write(f"{ROOT}/samples/lux_clone_test.wav", wav, 48000)
