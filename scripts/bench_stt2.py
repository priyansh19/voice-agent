"""Profile Granite TurboCTC on CPU: feature extraction vs forward, thread counts, int8 dynamic quant. Then Kokoro int8."""
import time, os, sys, copy
import numpy as np, soundfile as sf, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wav16, _ = sf.read(f"{ROOT}/samples/test_utterance_16k.wav", dtype="float32")
from transformers import AutoModelForCTC, AutoProcessor
mid = "ibm-granite/granite-speech-5.0-470m-turboctc"
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForCTC.from_pretrained(mid, dtype=torch.float32).eval()
print("model class:", type(model).__name__, " params(M):", sum(p.numel() for p in model.parameters())//1_000_000)
print("generation_config:", model.generation_config)

def run(m, a, n=3):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); inputs = proc([a], sampling_rate=16000); t1=time.perf_counter()
        with torch.inference_mode(): out = m.generate(**inputs)
        t2=time.perf_counter(); txt = proc.batch_decode(out, skip_special_tokens=True)[0]; t3=time.perf_counter()
        ts.append((t1-t0, t2-t1, t3-t2))
    f,g,d = min(ts, key=lambda x: x[1])
    return f*1000, g*1000, d*1000, txt

# direct forward instead of generate?
inputs = proc([wav16], sampling_rate=16000)
print("input keys:", list(inputs.keys()), {k: tuple(v.shape) for k,v in inputs.items() if hasattr(v,'shape')})
with torch.inference_mode():
    t=time.perf_counter(); o = model(**inputs); print(f"raw forward: {(time.perf_counter()-t)*1000:.0f}ms logits {tuple(o.logits.shape)}")
    t=time.perf_counter(); ids = o.logits.argmax(-1); print("argmax decode:", proc.batch_decode(ids, skip_special_tokens=True)[0][:60], f"{(time.perf_counter()-t)*1000:.0f}ms")

for th in (4, 6, 8, 12, 16):
    torch.set_num_threads(th)
    run(model, wav16, 1)
    f,g,d,txt = run(model, wav16)
    print(f"threads={th}: feat {f:.0f}ms  generate {g:.0f}ms  decode {d:.0f}ms")
torch.set_num_threads(8)
for sec in (0.5, 1, 2, 3):
    f,g,d,txt = run(model, wav16[:int(sec*16000)])
    print(f"{sec}s audio: feat {f:.0f}ms generate {g:.0f}ms -> {txt!r}")

# int8 dynamic quant on Linear
qm = torch.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
run(qm, wav16, 1)
f,g,d,txt = run(qm, wav16)
print(f"int8 dyn: generate {g:.0f}ms -> {txt!r}")

# --- Kokoro int8 + threads ---
import urllib.request
p8 = f"{ROOT}/models/kokoro/kokoro-v1.0.int8.onnx"
if not os.path.exists(p8):
    urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx", p8)
from kokoro_onnx import Kokoro
import onnxruntime as ort
print("ort providers:", ort.get_available_providers())
for name, path in (("fp32", f"{ROOT}/models/kokoro/kokoro-v1.0.onnx"), ("int8", p8)):
    kk = Kokoro(path, f"{ROOT}/models/kokoro/voices-v1.0.bin")
    kk.create("Warm up sentence here.", voice="am_adam", lang="en-us")
    for s in ["Sure, I can help with that.", "Tomorrow in Milan it should be mostly cloudy with a chance of light rain."]:
        t=time.perf_counter(); w, sr = kk.create(s, voice="am_adam", lang="en-us"); dt=time.perf_counter()-t
        print(f"[kokoro {name}] {dt*1000:.0f}ms for {len(w)/sr:.2f}s (RTF {dt/(len(w)/sr):.2f})")
