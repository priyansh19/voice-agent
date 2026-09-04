"""Quality check of the bucket/text-padding approach: write wavs for exact-dynamic (ORT), bucketed (ORT), bucketed (GPU f32)."""
import sys, os, time, soundfile as sf, numpy as np
sys.path.insert(0, os.path.join(ROOT, "tts_worker")); sys.path.insert(0, ROOT)
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
from lux_fast import LuxFast
ref = f"{ROOT}/samples/test_utterance_24k.wav"
texts = {"short": "Sure, I can help with that.", "mid": "Yes, bring an umbrella just in case.",
         "long": "Tomorrow in Milan it should be mostly cloudy with a chance of light rain in the afternoon."}
mode = sys.argv[1]
if mode == "ort":
    lf = LuxFast(threads=6, compute="ort"); lf.encode_prompt(ref, duration=3.0)
    for k, t in texts.items():
        tm = {}; w = lf.synth(t, timings=tm); sf.write(f"{ROOT}/samples/bk_exact_{k}.wav", w, 48000); print("exact", k, {a: round(b) if isinstance(b, float) else b for a, b in tm.items()})
    lf.compute = "ov"; lf.buckets = [384, 448, 512, 640, 768]
    import openvino as ov; lf.core = ov.Core(); lf.core.set_property({"CACHE_DIR": f"{ROOT}/models/lux_ov_cache"}); lf.ov_device = "CPU"
    from lux_fast import _find_onnx; lf.fm_path = _find_onnx("fm_decoder.onnx") or _find_onnx("*fm*decoder*.onnx")
    for k, t in texts.items():
        tm = {}; w = lf.synth(t, timings=tm); sf.write(f"{ROOT}/samples/bk_cpu_{k}.wav", w, 48000); print("bucket-cpu", k, {a: round(b) if isinstance(b, float) else b for a, b in tm.items()})
else:
    lf = LuxFast(threads=6, compute="ov", ov_device="GPU", buckets=[384, 448, 512, 640, 768], cache_dir=f"{ROOT}/models/lux_ov_cache", precision=sys.argv[2] if len(sys.argv) > 2 else "f32")
    lf.encode_prompt(ref, duration=3.0)
    t0 = time.perf_counter(); lf.prepare(); print(f"prepare {time.perf_counter()-t0:.0f}s")
    lf.synth("This is a warm up sentence.")
    for k, t in texts.items():
        for steps in (4, 2):
            tm = {}; t1 = time.perf_counter(); w = lf.synth(t, steps=steps, timings=tm)
            print(f"bucket-gpu {k} steps={steps}", {a: round(b) if isinstance(b, float) else b for a, b in tm.items()}, f"audio {len(w)/48000:.2f}s")
            sf.write(f"{ROOT}/samples/bk_gpu{steps}_{k}.wav", w, 48000)
