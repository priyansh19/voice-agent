import sys, os, time, soundfile as sf
sys.path.insert(0, os.path.join(ROOT, "tts_worker")); sys.path.insert(0, ROOT)
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
from lux_fast import LuxFast
ref = f"{ROOT}/samples/test_utterance_24k.wav"
texts = ["Sure.", "Sure, I can help with that.", "Tomorrow in Milan it should be mostly cloudy with a chance of light rain in the afternoon."]
def bench(tag, lf, dur, steps=4):
    t=time.perf_counter(); lf.encode_prompt(ref, duration=dur); enc_ms=(time.perf_counter()-t)*1000
    lf.synth("This is a warm up sentence for the model.", steps=steps)
    for text in texts:
        tm={}; t=time.perf_counter(); wav = lf.synth(text, steps=steps, timings=tm); tot=(time.perf_counter()-t)*1000
        print(f"[{tag} prompt={dur}s steps={steps}] {tot:5.0f}ms (flow {tm['flow_ms']:.0f} voc {tm['vocoder_ms']:.0f}) -> {len(wav)/48000:.2f}s audio  {text[:30]!r}")
        sf.write(f"{ROOT}/samples/lux_{tag}_{len(text)}.wav", wav, 48000)
    print(f"[{tag}] encode_prompt {enc_ms:.0f}ms")
mode = sys.argv[1]
if mode == "ort":
    for th in (4, 8, 12):
        lf = LuxFast(threads=th, backend="ort")
        for dur in (3, 5): bench(f"ort{th}", lf, dur)
        if th == 8: bench("ort8", lf, 3, steps=2)
else:
    dev = sys.argv[2]
    lf = LuxFast(threads=8, backend="ov", ov_device=dev, cache_dir=f"{ROOT}/models/lux_ov_cache")
    for dur in (3, 5): bench(f"ov-{dev}", lf, dur)
    bench(f"ov-{dev}", lf, 3, steps=2)
