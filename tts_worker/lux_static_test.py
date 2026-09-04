"""Does padding the flow-decoder inputs to a static bucket change the output? And how fast is a static bucket on GPU/NPU?"""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from lux_fast import LuxFast, _find_onnx
from zipvoice.models.modules.solver import get_time_steps
import openvino as ov
lf = LuxFast(threads=8, backend="ort"); enc = lf.encode_prompt(f"{ROOT}/samples/test_utterance_24k.wav", duration=3)
pt, pfl, pf, prms = enc["prompt_tokens"], enc["prompt_features_lens"], enc["prompt_features"], enc["prompt_rms"]
m = lf.tts.model; D = m.feat_dim; P = pf.shape[1]; print("prompt frames", P, "feat_dim", D)

def prep(text, speed=1.0):
    tokens = lf.tts.tokenizer.texts_to_token_ids([text]); Tt, Tp = len(tokens[0]), len(pt[0])
    eff = (Tt + Tp) / (Tp + Tt / speed)
    t = time.perf_counter()
    tc = m.run_text_encoder(torch.tensor(tokens), torch.tensor(pt), torch.tensor(P), torch.tensor(eff, dtype=torch.float32))
    return tc, (time.perf_counter() - t) * 1000

def flow(run_fm, tc, N, x_full, steps=4, guidance=3.0, t_shift=0.5):
    n = tc.shape[1]
    tc = torch.nn.functional.pad(tc, (0, 0, 0, N - n)); x = x_full[:, :N].clone()
    sc = torch.nn.functional.pad(pf, (0, 0, 0, N - P))
    ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=steps, t_shift=t_shift); g = torch.tensor(guidance, dtype=torch.float32)
    for s in range(steps):
        tc_, tn = ts[s], ts[s + 1]
        v = run_fm(tc_, x, tc, sc, g)
        x1, x0 = x + (1.0 - tc_) * v, x - tc_ * v
        x = (1.0 - tn) * x0 + tn * x1 if s < steps - 1 else x1
    return x[:, P:n]

texts = ["Sure, I can help with that.", "Tomorrow in Milan it should be mostly cloudy with a chance of light rain in the afternoon."]
torch.manual_seed(0); x_full = torch.randn(1, 1024, D)
for text in texts:
    tc, te_ms = prep(text); n = tc.shape[1]; N = int(np.ceil(n / 64) * 64)
    ref = flow(m.run_fm_decoder, tc, n, x_full)          # dynamic (exact) on ORT CPU
    padded = flow(m.run_fm_decoder, tc, N, x_full)       # padded to bucket, ORT CPU
    err = ((ref - padded) ** 2).mean() / (ref ** 2).mean()
    print(f"{text[:25]!r}: text_enc {te_ms:.0f}ms frames={n} bucket={N}  rel_mse(pad vs exact)={err:.2e}  corr={np.corrcoef(ref.flatten(), padded.flatten())[0,1]:.4f}")

# static compile of fm_decoder per bucket on GPU / NPU
core = ov.Core(); core.set_property({"CACHE_DIR": f"{ROOT}/models/lux_ov_cache"})
fm_path = _find_onnx("fm_decoder.onnx") or _find_onnx("*fm*decoder*.onnx"); base = core.read_model(fm_path)
print("fm inputs:", [(i.get_any_name(), i.get_partial_shape()) for i in base.inputs])
for dev in ["GPU", "NPU"]:
    for N in (384, 512):
        try:
            mdl = core.read_model(fm_path)
            mdl.reshape({0: [], 1: [1, N, D], 2: [1, N, mdl.inputs[2].get_partial_shape()[2].get_length()], 3: [1, N, D], 4: []})
            t = time.perf_counter(); cm = core.compile_model(mdl, dev, {"PERFORMANCE_HINT": "LATENCY"}); req = cm.create_infer_request()
            print(f"[{dev} N={N}] compile {time.perf_counter()-t:.1f}s")
            names = [i.get_any_name() for i in cm.inputs]
            def run_fm(t_, x, tcnd, sc, g):
                req.infer(dict(zip(names, [t_.numpy(), x.numpy(), tcnd.numpy(), sc.numpy(), g.numpy()])))
                return torch.from_numpy(np.array(req.get_output_tensor(0).data))
            tc, _ = prep(texts[0]); n = tc.shape[1]
            flow(run_fm, tc, N, x_full)  # warm
            for steps in (4, 2):
                t = time.perf_counter(); out = flow(run_fm, tc, N, x_full, steps=steps); dt = (time.perf_counter() - t) * 1000
                ref = flow(m.run_fm_decoder, tc, n, x_full, steps=steps)
                err = ((ref - out) ** 2).mean() / (ref ** 2).mean()
                print(f"[{dev} N={N}] steps={steps}: {dt:.0f}ms ({dt/steps:.0f}ms/step) rel_mse vs ORT exact={err:.2e}")
        except Exception as e:
            print(f"[{dev} N={N}] FAILED {type(e).__name__}: {str(e)[:200]}")
