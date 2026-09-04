"""Simulate the hosted page: stream a wav as int16 16 kHz frames to /audio in real time, collect the reply PCM."""
import os, sys
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
os.chdir(ROOT)
import asyncio, json, time, numpy as np, soundfile as sf, websockets
sys.path.insert(0, ROOT)
from agent.audio_io import resample
async def main(path):
    a, sr = sf.read(path, dtype="float32"); a = resample(a, sr, 16000)
    a = np.concatenate([a, np.zeros(16000, np.float32)]); pcm = (a * 32767).astype(np.int16)
    got = bytearray(); events = []
    async with websockets.connect("ws://127.0.0.1:8765/ws") as ctl, websockets.connect("ws://127.0.0.1:8765/audio", max_size=None) as aud:
        await ctl.send(json.dumps({"cmd": "mute", "value": False}))
        async def reader():
            async for m in aud:
                if isinstance(m, bytes): got.extend(m)
                else: events.append(m)
        async def ctl_reader():
            async for m in ctl:
                d = json.loads(m)
                if d["kind"] in ("transcript", "response", "metrics"): events.append((d["kind"], d["data"] if d["kind"] != "metrics" else d["data"]["first_audio_ms"]))
        rt, ct = asyncio.create_task(reader()), asyncio.create_task(ctl_reader())
        t0 = time.perf_counter()
        for i in range(0, len(pcm) - 511, 512):
            while time.perf_counter() - t0 < i / 16000: await asyncio.sleep(0.002)
            await aud.send(pcm[i:i + 512].tobytes())
        await asyncio.sleep(12)
        await ctl.send(json.dumps({"cmd": "mute", "value": True}))
        rt.cancel(); ct.cancel()
    print(f"received {len(got)/2/48000:.1f}s of reply audio over the socket")
    for e in events: print("  ", e)
asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "samples/test_utterance_16k.wav"))
