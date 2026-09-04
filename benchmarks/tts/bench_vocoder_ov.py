"""Can the LuxTTS vocoder (LinaCodec Vocos, torch CPU ~100-300 ms) run on the Intel GPU via OpenVINO with static shapes?"""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.join(ROOT, "tts_worker")); sys.path.insert(0, ROOT)
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
from lux_fast import LuxFast
import openvino as ov
lf = LuxFast(threads=6, compute="ort")
lf.encode_prompt(f"{ROOT}/voices/me.wav", duration=3.0)
voc = lf.tts.vocos.eval()
D = lf.D

class Wrap(torch.nn.Module):
    def __init__(s, v): super().__init__(); s.v = v
    def forward(s, feats): return s.v.decode(feats)

core = ov.Core(); core.set_property({"CACHE_DIR": f"{ROOT}/models/lux_ov_cache"})
for frames in (128, 256, 384):
    x = torch.randn(1, D, frames)
    with torch.inference_mode():
        for _ in range(2): ref = voc.decode(x)
        t = time.perf_counter(); ref = voc.decode(x); cpu_ms = (time.perf_counter() - t) * 1000
    try:
        t = time.perf_counter()
        m = ov.convert_model(Wrap(voc), example_input=x, input=[("feats", [1, D, frames])])
        conv_ms = (time.perf_counter() - t) * 1000
        for dev in ("GPU",):
            t = time.perf_counter(); cm = core.compile_model(m, dev, {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}); req = cm.create_infer_request()
            comp = time.perf_counter() - t
            req.infer({"feats": x.numpy()})
            ts = []
            for _ in range(3):
                t = time.perf_counter(); req.infer({"feats": x.numpy()}); ts.append((time.perf_counter() - t) * 1000)
            out = np.array(req.get_output_tensor(0).data)
            err = float(np.abs(out - ref.numpy()).mean() / (np.abs(ref.numpy()).mean() + 1e-9))
            print(f"frames={frames} ({frames/93.75:.1f}s audio): CPU torch {cpu_ms:.0f}ms | OV {dev} {min(ts):.0f}ms (compile {comp:.0f}s, convert {conv_ms:.0f}ms) rel_err {err:.3f} nan={np.isnan(out).any()}", flush=True)
    except Exception as e:
        print(f"frames={frames}: CPU {cpu_ms:.0f}ms | OV FAILED {e.__class__.__name__}: {str(e)[:200]}", flush=True)
