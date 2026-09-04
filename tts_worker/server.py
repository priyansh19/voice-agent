"""TTS worker process (separate venv: LuxTTS needs transformers<=4.57, the STT needs >=5.16).

Protocol over stdio (binary):  request = one JSON line on stdin;  reply = one JSON header line + raw float32 PCM.
  {"cmd":"synth","id":N,"text":"..."}      -> {"id":N,"samples":S,"sr":48000,...} + S*4 bytes
  {"cmd":"fillers","texts":[...]}          -> one {"filler":text,"samples":S} + bytes per text
  {"cmd":"ping"}                           -> {"pong":true}
All logging goes to stderr; fd 1 is reserved for the protocol.
"""
import os, sys, json, time, hashlib, argparse

proto_fd = os.dup(1)
os.dup2(2, 1)                       # anything a library prints to stdout now lands on stderr
sys.stdout = sys.stderr
proto = os.fdopen(proto_fd, "wb", buffering=0)
try: sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch, yaml
from lux_fast import LuxFast


def log(*a):
    print("[tts-worker]", *a, file=sys.stderr, flush=True)


def send(header: dict, pcm: np.ndarray | None = None):
    if pcm is not None:
        header["samples"] = int(len(pcm))
    proto.write((json.dumps(header) + "\n").encode("utf-8"))
    if pcm is not None and len(pcm):
        proto.write(pcm.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--prepare", action="store_true", help="compile all buckets then exit")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["tts"]
    ref = cfg["voice_ref"]; ref = ref if os.path.isabs(ref) else os.path.join(ROOT, ref)
    if not os.path.exists(ref):
        log(f"voice reference {ref} not found — record one with: python main.py --record-voice"); send({"error": "no voice_ref"}); return

    t = time.perf_counter()
    lux = LuxFast(threads=cfg.get("threads", 6), compute=cfg.get("compute", "ov"), ov_device=cfg.get("ov_device", "GPU"),
                  buckets=cfg.get("buckets"), cache_dir=os.path.join(ROOT, "models", "lux_ov_cache"), log=log)
    log(f"model loaded in {time.perf_counter()-t:.1f}s")

    # voice prompt encoding is slow (runs an ASR on the reference); cache it on disk
    dur = float(cfg.get("prompt_duration_s", 3.0))
    key = hashlib.md5(open(ref, "rb").read()).hexdigest()[:10] + f"_{dur:g}_v2"
    cache = os.path.join(ROOT, "models", "voice_cache", f"{key}.pt"); os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        lux.load_encoded(torch.load(cache, weights_only=False)); log(f"voice prompt loaded from cache ({key})")
    else:
        # the clone conditions on the first `dur` seconds: trim lead-in silence so those seconds are all voice
        import soundfile as sf
        a, sr = sf.read(ref, dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        # condense: keep voiced 20 ms frames plus 0.2 s of context, drop long pauses, so the first `dur` s are speech
        fr = int(0.02 * sr); n = len(a) // fr
        r = np.sqrt((a[: n * fr].reshape(-1, fr) ** 2).mean(axis=1))
        keep = r >= max(0.004, 0.05 * float(r.max()))
        pad = 10                                              # 200 ms context on each side of speech
        idx = np.nonzero(keep)[0]
        mask = np.zeros(n, dtype=bool)
        for i in idx: mask[max(0, i - pad): i + pad + 1] = True
        cond = a[: n * fr].reshape(-1, fr)[mask].reshape(-1)
        trimmed = os.path.join(os.path.dirname(cache), f"{key}_trim.wav"); sf.write(trimmed, cond, sr)
        log(f"reference condensed {len(a)/sr:.1f}s -> {len(cond)/sr:.1f}s of speech for prompt encoding")
        t = time.perf_counter(); enc = lux.encode_prompt(trimmed, duration=dur); torch.save(enc, cache)
        log(f"voice prompt encoded in {time.perf_counter()-t:.1f}s -> {cache}")
    log(f"prompt frames={lux.P} tokens={lux.Tp}")

    t = time.perf_counter(); lux.prepare(); log(f"buckets ready in {time.perf_counter()-t:.1f}s")
    if args.prepare:
        send({"event": "prepared"}); return
    tm = {}; lux.synth("This is a warm up sentence for the speech model.", steps=cfg.get("steps", 4), timings=tm)
    log(f"warm synth {tm}")
    send({"event": "ready", "prompt_frames": lux.P, "buckets": lux.buckets})

    steps, speed = int(cfg.get("steps", 4)), float(cfg.get("speed", 1.0))
    for line in sys.stdin.buffer:
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        cmd = req.get("cmd")
        if cmd == "ping":
            send({"pong": True})
        elif cmd == "synth":
            tm = {}
            try:
                wav = lux.synth(req["text"], steps=req.get("steps", steps), speed=speed, timings=tm)
                send({"id": req["id"], "sr": 48000, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in tm.items()}}, wav)
            except Exception as e:
                log(f"synth failed for {req['text']!r}: {e}")
                send({"id": req["id"], "sr": 48000, "error": str(e)}, np.zeros(0, dtype=np.float32))
        elif cmd == "fillers":
            for text in req.get("texts", []):
                try:
                    wav = lux.synth(text, steps=steps, speed=speed)
                except Exception as e:
                    log(f"filler failed {text!r}: {e}"); continue
                send({"filler": text, "sr": 48000}, wav)
            send({"fillers_done": True})
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
