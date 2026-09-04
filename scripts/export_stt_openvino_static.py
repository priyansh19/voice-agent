"""Static-shape OpenVINO export of Granite TurboCTC (bucketed lengths) for Intel GPU / NPU."""
import time, os, sys
import numpy as np, soundfile as sf, torch, openvino as ov
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/models/granite_stt_ov"; os.makedirs(OUT, exist_ok=True)
from transformers import AutoModelForCTC, AutoProcessor
mid = "ibm-granite/granite-speech-5.0-470m-turboctc"
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForCTC.from_pretrained(mid, dtype=torch.float32).eval()
wav16, _ = sf.read(f"{ROOT}/samples/test_utterance_16k.wav", dtype="float32")
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, input_features, attention_mask):
        return s.m(input_features=input_features, attention_mask=attention_mask).logits
FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300   # 50 fps -> 300 frames = 6 s bucket
xml = f"{OUT}/model_static{FRAMES}.xml"
if not os.path.exists(xml):
    a = np.zeros(int(FRAMES/50*16000), dtype=np.float32); a[:len(wav16)] = wav16[:len(a)]
    ex = proc([a], sampling_rate=16000); print("example shapes", {k: tuple(v.shape) for k,v in ex.items()})
    t = time.perf_counter()
    ovm = ov.convert_model(Wrap(model), example_input=dict(ex),
                           input=[("input_features", [1, ex["input_features"].shape[1], 320]), ("attention_mask", [1, ex["attention_mask"].shape[1]])])
    ov.save_model(ovm, xml); print(f"converted static {FRAMES} in {time.perf_counter()-t:.1f}s")
core = ov.Core(); core.set_property({"CACHE_DIR": f"{OUT}/cache"})
m = core.read_model(xml); T = m.inputs[0].get_partial_shape()[1].get_length()
def feats(a):
    buf = np.zeros(int(T/50*16000), dtype=np.float32); n=min(len(a), len(buf)); buf[:n] = a[:n]
    inp = proc([buf], sampling_rate=16000)
    mask = np.zeros_like(inp["attention_mask"].numpy()); mask[:, :int(np.ceil(n/16000*50))] = 1   # mask only real audio
    return {"input_features": inp["input_features"].numpy(), "attention_mask": mask}, inp["attention_mask"].numpy()
for dev in ["GPU", "NPU", "CPU"]:
    if dev not in core.available_devices: continue
    try:
        t = time.perf_counter(); cm = core.compile_model(m, dev, {"PERFORMANCE_HINT": "LATENCY"})
        print(f"[{dev}] compile {time.perf_counter()-t:.1f}s"); req = cm.create_infer_request()
        for sec in (1, 2, 3, 5.23):
            a = wav16[:int(sec*16000)]
            for maskmode in ("masked", "full"):
                feed, fullmask = feats(a)
                if maskmode == "full": feed = dict(feed, attention_mask=fullmask)
                req.infer(feed)
                ts=[]
                for _ in range(3):
                    t=time.perf_counter(); req.infer(feed); ts.append(time.perf_counter()-t)
                logits = req.get_output_tensor(0).data
                txt = proc.batch_decode(torch.from_numpy(np.array(logits)).argmax(-1), skip_special_tokens=True)[0]
                print(f"[{dev}] {sec}s audio ({maskmode}): {min(ts)*1000:.0f}ms -> {txt!r}")
    except Exception as e:
        print(f"[{dev}] FAILED: {type(e).__name__}: {str(e)[:250]}")
