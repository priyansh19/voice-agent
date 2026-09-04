import sys, time, glob, os, numpy as np, soundfile as sf
import os, sys
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import openvino_genai as ov_genai
sys.stdout.reconfigure(encoding="utf-8")
def path(repo): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
wavs = {"en": "samples/test_utterance_16k.wav", "hi": "samples/hindi_16k.wav", "hinglish": "samples/hinglish_16k.wav"}
for repo in sys.argv[2:] or ["OpenVINO/whisper-large-v3-turbo-int4-ov", "OpenVINO/whisper-small-int8-ov"]:
    for dev in sys.argv[1].split(","):
        try:
            t = time.perf_counter(); pipe = ov_genai.WhisperPipeline(path(repo), dev, CACHE_DIR="models/whisper_ov_cache")
            cfg = pipe.get_generation_config(); cfg.max_new_tokens = 96; cfg.task = "transcribe"; cfg.return_timestamps = False
            print(f"[{repo.split('/')[1]} @ {dev}] load {time.perf_counter()-t:.1f}s", flush=True)
            a, _ = sf.read(wavs["en"], dtype="float32"); pipe.generate(a.tolist(), cfg)   # warm
            for name, w in wavs.items():
                a, _ = sf.read(w, dtype="float32")
                ts = []
                for _ in range(2):
                    t = time.perf_counter(); r = pipe.generate(a.tolist(), cfg); ts.append(time.perf_counter() - t)
                print(f"   {name:8s} {min(ts)*1000:5.0f}ms -> {r.texts[0].strip()!r}", flush=True)
        except Exception as e:
            print(f"[{repo.split('/')[1]} @ {dev}] FAILED {e.__class__.__name__}: {str(e)[:160]}", flush=True)
