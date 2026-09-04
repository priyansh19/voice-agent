"""Export Granite Speech 5.0 TurboCTC to OpenVINO IR and time it on CPU / Intel Arc GPU / NPU."""
import time, os, sys
import numpy as np, soundfile as sf, torch, openvino as ov
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/models/granite_stt_ov"; os.makedirs(OUT, exist_ok=True)
from transformers import AutoModelForCTC, AutoProcessor
mid = "ibm-granite/granite-speech-5.0-470m-turboctc"
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForCTC.from_pretrained(mid, dtype=torch.float32).eval()
wav16, _ = sf.read(f"{ROOT}/samples/test_utterance_16k.wav", dtype="float32")
inputs = proc([wav16], sampling_rate=16000)
print({k:(tuple(v.shape), v.dtype) for k,v in inputs.items()})

class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, input_features, attention_mask):
        return s.m(input_features=input_features, attention_mask=attention_mask).logits

xml = f"{OUT}/model.xml"
if not os.path.exists(xml):
    t = time.perf_counter()
    ovm = ov.convert_model(Wrap(model), example_input=dict(inputs),
                           input=[("input_features", [1, -1, 320]), ("attention_mask", [1, -1])])
    ov.save_model(ovm, xml)  # fp16 weights compressed by default
    print(f"converted in {time.perf_counter()-t:.1f}s")
core = ov.Core(); print("devices:", core.available_devices)
ref = proc.batch_decode(model(**inputs).logits.argmax(-1), skip_special_tokens=True)[0]
print("torch ref:", ref)
for dev in ["CPU", "GPU"]:
    if dev not in core.available_devices: continue
    try:
        t = time.perf_counter()
        cm = core.compile_model(xml, dev, {"PERFORMANCE_HINT": "LATENCY"})
        print(f"[{dev}] compile {time.perf_counter()-t:.1f}s")
        req = cm.create_infer_request()
        def run(a):
            inp = proc([a], sampling_rate=16000)
            feed = {"input_features": inp["input_features"].numpy(), "attention_mask": inp["attention_mask"].numpy()}
            req.infer(feed); return req.get_output_tensor(0).data
        run(wav16[:16000]); run(wav16)
        for sec in (0.5, 1, 2, 3, 5.23):
            a = wav16[:int(sec*16000)]
            ts = []
            for _ in range(3):
                t = time.perf_counter(); logits = run(a); ts.append(time.perf_counter()-t)
            txt = proc.batch_decode(torch.from_numpy(np.array(logits)).argmax(-1), skip_special_tokens=True)[0]
            print(f"[{dev}] {sec}s audio: {min(ts)*1000:.0f}ms -> {txt!r}")
    except Exception as e:
        print(f"[{dev}] FAILED: {type(e).__name__}: {str(e)[:300]}")
