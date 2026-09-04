import time, os, sys, torch, numpy as np, soundfile as sf
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
from zipvoice.luxvoice import LuxTTS
from zipvoice.onnx_modeling import sample
tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=4)
enc = tts.encode_prompt(f"{ROOT}/samples/test_utterance_24k.wav", duration=5, rms=0.01)
pt, pfl, pf, prms = enc.values()
print("prompt_tokens len:", len(pt[0]), "prompt_features:", tuple(pf.shape), "rms:", float(prms))
print("vocos hop/sr attrs:", {k: getattr(tts.vocos, k) for k in dir(tts.vocos) if k in ('hop_length','sample_rate','freq_range','return_48k')})
MIN_FRAMES = 48
def gen(text, steps=4, min_frames=MIN_FRAMES):
    tokens = tts.tokenizer.texts_to_token_ids([text])
    feats = sample(model=tts.model, tokens=tokens, prompt_tokens=pt, prompt_features=pf, speed=1.0*1.3, t_shift=0.5, guidance_scale=3.0, num_step=steps)
    n = feats.shape[1]
    feats = feats.permute(0, 2, 1) / 0.1
    if n < min_frames:  # pad time axis with zeros (silence) so vocoder convs have enough context, trim after
        feats = torch.nn.functional.pad(feats, (0, min_frames - n))
    wav = tts.vocos.decode(feats).squeeze(1).clamp(-1, 1)
    if n < min_frames:
        wav = wav[:, : int(wav.shape[1] * n / min_frames)]
    if prms < 0.1: wav = wav * (prms / 0.1)
    return wav, n
for text in ["Sure.", "Hmm.", "Okay, sure.", "Yes, bring an umbrella.", "Sure, I can help with that.",
             "Tomorrow in Milan it should be mostly cloudy with a chance of light rain in the afternoon."]:
    try:
        t=time.perf_counter(); wav, n = gen(text); dt=time.perf_counter()-t
        print(f"OK  {dt*1000:5.0f}ms frames={n:3d} audio={wav.shape[1]/48000:.2f}s  {text!r}")
        sf.write(f"{ROOT}/samples/lux_{n}.wav", wav.squeeze().numpy(), 48000)
    except Exception as e:
        print(f"FAIL frames=? {type(e).__name__}: {str(e)[:70]}  {text!r}")
# steps=2 vs 4 timing on a mid sentence
for steps in (1,2,3,4):
    t=time.perf_counter(); wav,n = gen("Sure, I can help with that.", steps); print(f"steps={steps}: {(time.perf_counter()-t)*1000:.0f}ms")
