"""Microphone capture (16 kHz mono frames) and gapless speaker playback (48 kHz) with instant stop for barge-in."""
import threading, queue, time
import numpy as np
import sounddevice as sd


class Microphone:
    def __init__(self, sample_rate=16000, frame_samples=512, device=None):
        self.sr, self.n, self.device = sample_rate, frame_samples, device
        self.q = queue.Queue(maxsize=400)
        self.stream = None

    def _cb(self, indata, frames, t, status):
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def start(self):
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=self.n,
                                     device=self.device, callback=self._cb)
        self.stream.start()

    def frames(self):
        while True:
            yield self.q.get()

    def stop(self):
        if self.stream:
            self.stream.stop(); self.stream.close()


class Speaker:
    """Pull-based output stream fed from an ordered list of chunks. Each chunk may carry an on_start callback
    that fires when its first sample reaches the device. stop() flushes immediately (barge-in)."""
    def __init__(self, sample_rate=48000, device=None, sink=None):
        self.sr, self.device = sample_rate, device
        self.sink = sink                       # optional callable(np.ndarray): test mode, no audio device
        self.remote = None                     # optional callable(int16 bytes): browser playback over WebSocket
        self.remote_stop = None                # optional callable(): tell the browser to flush its queue
        self.remote_busy_until = 0.0
        self.lock = threading.Lock()
        self.chunks = []                       # [(np.ndarray, callback|None)]
        self.pos = 0
        self.playing = False
        self.last_active = 0.0
        self.stream = None

    def start(self):
        if self.sink is not None:
            return
        try:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=480,
                                          device=self.device, callback=self._cb, latency="low")
            self.stream.start()
        except Exception as e:                 # headless machine / CI: keep running, audio goes to browsers only
            print(f"[audio] no output device ({e.__class__.__name__}); local playback disabled", flush=True)
            self.stream = None
            self.sink = lambda x: None

    def _cb(self, outdata, frames, t, status):
        out = np.zeros(frames, dtype=np.float32); filled = 0; fire = []
        with self.lock:
            while filled < frames and self.chunks:
                c, cb = self.chunks[0]
                if self.pos == 0 and cb:
                    fire.append(cb)
                take = min(frames - filled, len(c) - self.pos)
                out[filled:filled + take] = c[self.pos:self.pos + take]; filled += take; self.pos += take
                if self.pos >= len(c):
                    self.chunks.pop(0); self.pos = 0
            self.playing = filled > 0
            if self.playing:
                self.last_active = time.perf_counter()
        for cb in fire:
            cb()
        outdata[:, 0] = out

    def play(self, chunk: np.ndarray, on_start=None):
        if self.remote is not None:            # browser playback (takes precedence): ship int16 PCM, track when it ends
            now = time.perf_counter()
            self.remote_busy_until = max(now, self.remote_busy_until) + len(chunk) / self.sr
            self.remote((np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes())
            self.last_active = now
            if on_start: on_start()
            return
        if self.sink is not None:
            if on_start: on_start()
            self.sink(chunk); self.last_active = time.perf_counter(); return
        with self.lock:
            self.chunks.append((chunk.astype(np.float32, copy=False), on_start))

    def stop(self):
        with self.lock:
            self.chunks.clear(); self.pos = 0; self.playing = False
        if self.remote is not None:
            self.remote_busy_until = 0.0
            if self.remote_stop: self.remote_stop()

    def is_busy(self, tail_ms=450):
        if self.remote is not None:
            return time.perf_counter() < self.remote_busy_until + tail_ms / 1000
        if self.sink is not None:
            return False
        return self.playing or bool(self.chunks) or (time.perf_counter() - self.last_active) * 1000 < tail_ms

    def wait_idle(self, timeout=120):
        t0 = time.perf_counter()
        if self.remote is not None:
            while time.perf_counter() < self.remote_busy_until and time.perf_counter() - t0 < timeout:
                time.sleep(0.02)
            return
        if self.sink is not None:
            return
        while (self.chunks or self.playing) and time.perf_counter() - t0 < timeout:
            time.sleep(0.01)

    def close(self):
        if self.stream:
            self.stream.stop(); self.stream.close()


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)
