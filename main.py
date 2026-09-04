"""Sub-second local voice agent on open-weight models.

  python main.py                      live conversation from the microphone
  python main.py --record-voice 12    record a 12 s reference of your voice to voices/me.wav
  python main.py --wav samples/x.wav  offline test: run one utterance from a file (no mic), save reply to out/
  python main.py --text "..."         skip STT: measure LLM + TTS only
  python main.py --prepare            export / compile all models (first run), then exit
  python main.py --list-devices
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
from agent import config as C


def log(*a):
    print(*a, flush=True)


def record_voice(cfg, seconds, path):
    import sounddevice as sd, soundfile as sf
    log(f"Recording {seconds}s in 2s... read something natural, e.g. a few sentences about your day.")
    time.sleep(2); log("Recording NOW")
    audio = sd.rec(int(seconds * 48000), samplerate=48000, channels=1, dtype="float32", device=cfg.audio.input_device)
    sd.wait()
    a = audio[:, 0]
    a = a / max(1e-6, np.abs(a).max()) * 0.9
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, a, 48000)
    log(f"saved {path}  (peak-normalized). Re-run --prepare to encode it.")


def load_wav16(path):
    import soundfile as sf
    from agent.audio_io import resample
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    return resample(a, sr, 16000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--record-voice", nargs="?", const=12, type=float, metavar="SECONDS")
    ap.add_argument("--wav", nargs="+", help="offline test utterance(s)")
    ap.add_argument("--text", nargs="+", help="offline test: text turns (no STT)")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-audio-out", action="store_true", help="write replies to out/ instead of the speaker")
    args = ap.parse_args()
    cfg = C.load(args.config)

    if args.list_devices:
        import sounddevice as sd; print(sd.query_devices()); return
    if args.record_voice:
        record_voice(cfg, args.record_voice, C.abspath(cfg.tts.voice_ref)); return

    sink_buf = []
    sink = (lambda x: sink_buf.append(x)) if (args.no_audio_out or args.wav or args.text) else None
    from agent.pipeline import VoiceAgent
    agent = VoiceAgent(cfg, log, sink=sink)
    agent.warmup()
    if args.prepare:
        log("prepared."); agent.close(); return

    def save_reply(tag):
        import soundfile as sf
        if sink_buf:
            os.makedirs(C.abspath("out"), exist_ok=True)
            p = C.abspath(f"out/reply_{tag}.wav"); sf.write(p, np.concatenate(sink_buf), 48000); sink_buf.clear(); log(f"[saved] {p}")

    try:
        if args.wav:
            n = cfg.audio.frame_samples
            for i, w in enumerate(args.wav):
                a = np.concatenate([load_wav16(w), np.zeros(16000, np.float32)])   # trailing silence closes the turn
                a = a[: len(a) // n * n]
                agent.feed(a.reshape(-1, n))
                agent.speaker.wait_idle()
                while agent.turn and not (agent.turn.m.audio_done or agent.turn.cancel.is_set()):
                    time.sleep(0.05)
                save_reply(i)
        elif args.text:
            for i, text in enumerate(args.text):
                t = agent._start_turn(None, speculative=False, t_tent=time.perf_counter(), text=text)
                t.m.speech_end = time.perf_counter(); agent._commit(t)
                while not (t.m.audio_done or t.cancel.is_set()):
                    time.sleep(0.05)
                save_reply(i)
        else:
            agent.run_mic()
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()


if __name__ == "__main__":
    main()
