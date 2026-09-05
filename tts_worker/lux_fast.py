"""LuxTTS (Apache-2.0 zero-shot voice cloning) tuned for low latency on Intel CPU + Arc iGPU.

Fixes / additions over the upstream CPU path:
  1. Duration formula: upstream multiplies `speed` by 1.3 inside a ratio that yields ZERO generated frames for
     short sentences.  We compute the speed argument so the generated length is the natural rate.
  2. Short-text vocoder crash: pad features to a minimum length before the vocoder and trim the audio.
  3. Static-shape buckets for the flow-matching decoder so it can run on the Intel GPU through OpenVINO
     (dynamic shapes recompile for ~20 s on every new length).  A bucket is filled with *real* filler text
     (not zeros) so the model stays in-distribution; the filler speech is cut off afterwards.
"""
import os, glob, time
import numpy as np
import torch
from zipvoice.luxvoice import LuxTTS
from zipvoice.models.modules.solver import get_time_steps

FPS = 93.75            # feature frames per second (24 kHz / 256 hop)
HOP48 = 512            # output samples per frame at 48 kHz
MIN_VOC_FRAMES = 48    # vocoder needs at least this many frames
PAD_TEXT = ", and so on and so forth" * 60   # filler used to fill a bucket; cut away after synthesis
FADE_MS = 25


def _find_onnx(name):
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--YatharthS--LuxTTS/snapshots/*/{name}"))
    return hits[0] if hits else None


class LuxFast:
    def __init__(self, threads=6, compute="ort", ov_device="GPU", buckets=None, cache_dir=None,
                 precision="f32", log=print):
        self.log = log
        self.compute = compute
        if compute == "mps":                       # Apple Silicon: LuxTTS's own torch path on Metal
            self.tts = LuxTTS("YatharthS/LuxTTS", device="mps")
            self.ort = None; self.D = 0
        else:
            self.tts = LuxTTS("YatharthS/LuxTTS", device="cpu", threads=threads)
            self.ort = self.tts.model
            self.D = self.ort.feat_dim
        torch.set_num_threads(threads)
        self.ov_device = ov_device
        self.buckets = sorted(buckets or [])
        self.precision = precision
        self.static = {}          # N -> (infer_request, input_names)
        self.enc = None
        self.P = self.Tp = 0
        self.pad_ids = None
        if compute == "ov":
            import openvino as ov
            self.core = ov.Core()
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                self.core.set_property({"CACHE_DIR": cache_dir})
            self.fm_path = _find_onnx("fm_decoder.onnx") or _find_onnx("*fm*decoder*.onnx")
            assert self.fm_path, "LuxTTS fm_decoder.onnx not found in the HF cache"

    # ------------------------------------------------------------------ setup
    def prepare(self):
        """Compile every bucket now (first time ~1 min each on GPU; cached on disk afterwards)."""
        if self.compute != "ov":
            return
        for N in self.buckets:
            self._compiled(N)

    def _compiled(self, N):
        if N in self.static:
            return self.static[N]
        t = time.perf_counter()
        m = self.core.read_model(self.fm_path)
        C = m.inputs[2].get_partial_shape()[2].get_length()
        m.reshape({0: [], 1: [1, N, self.D], 2: [1, N, C], 3: [1, N, self.D], 4: []})
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if self.ov_device == "GPU":
            cfg["INFERENCE_PRECISION_HINT"] = self.precision
        cm = self.core.compile_model(m, self.ov_device, cfg)
        req = cm.create_infer_request()
        names = [i.get_any_name() for i in cm.inputs]
        self.static[N] = (req, names)
        self.log(f"[lux] compiled bucket N={N} on {self.ov_device} in {time.perf_counter()-t:.1f}s")
        return self.static[N]

    def encode_prompt(self, wav_path, duration=3.0, rms=0.01):
        self.enc = self.tts.encode_prompt(wav_path, duration=duration, rms=rms)
        self._after_encode()
        return self.enc

    def load_encoded(self, enc):
        self.enc = enc
        self._after_encode()

    def _after_encode(self):
        self.P = int(self.enc["prompt_features"].shape[1])
        self.Tp = len(self.enc["prompt_tokens"][0])
        self.pad_ids = self.tts.tokenizer.texts_to_token_ids([PAD_TEXT])[0] if self.compute != "mps" else []

    # ------------------------------------------------------------------ pieces
    def _run_fm_ort(self, t, x, tc, sc, g):
        return self.ort.run_fm_decoder(t=t, x=x, text_condition=tc, speech_condition=sc, guidance_scale=g)

    def _run_fm_static(self, N):
        req, names = self._compiled(N)
        def run(t, x, tc, sc, g):
            req.infer(dict(zip(names, [t.numpy(), x.numpy(), tc.numpy(), sc.numpy(), g.numpy()], strict=False)))
            return torch.from_numpy(np.array(req.get_output_tensor(0).data))
        return run

    def _text_condition(self, ids, target_total):
        """Run the ONNX text encoder so that it produces exactly `target_total` frames (prompt + generated)."""
        pt = torch.tensor(self.enc["prompt_tokens"])
        tokens = torch.tensor([ids])
        T = len(ids)
        speed_arg = self.P * (T + self.Tp) / (self.Tp * target_total)
        tc = None
        for _ in range(4):
            tc = self.ort.run_text_encoder(tokens, pt, torch.tensor(self.P), torch.tensor(speed_arg, dtype=torch.float32))
            n = tc.shape[1]
            if n == target_total:
                break
            if n < target_total:            # a frame or two short: zero-pad (negligible)
                tc = torch.nn.functional.pad(tc, (0, 0, 0, target_total - n))
                break
            speed_arg *= n / target_total * 1.002   # too long: speak slightly faster and retry
        return tc

    def _flow(self, run_fm, tc, N, steps, guidance, t_shift):
        pf = self.enc["prompt_features"]
        x = torch.randn(1, N, self.D)
        sc = torch.nn.functional.pad(pf, (0, 0, 0, N - self.P))
        ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=steps, t_shift=t_shift)
        g = torch.tensor(guidance, dtype=torch.float32)
        for s in range(steps):
            t_cur, t_next = ts[s], ts[s + 1]
            v = run_fm(t_cur, x, tc, sc, g)
            x1, x0 = x + (1.0 - t_cur) * v, x - t_cur * v
            x = (1.0 - t_next) * x0 + t_next * x1 if s < steps - 1 else x1
        return x

    def _vocode(self, feats):
        n = feats.shape[1]
        feats = feats.permute(0, 2, 1) / 0.1
        if n < MIN_VOC_FRAMES:
            feats = torch.nn.functional.pad(feats, (0, MIN_VOC_FRAMES - n))
        wav = self.tts.vocos.decode(feats).squeeze(1).clamp(-1, 1)
        if n < MIN_VOC_FRAMES:
            wav = wav[:, : n * HOP48]
        prms = self.enc["prompt_rms"]
        if prms < 0.1:
            wav = wav * (prms / 0.1)
        return wav.squeeze(0).numpy().astype(np.float32)

    # ------------------------------------------------------------------ public
    def natural_frames(self, ids, speed=1.0):
        return self.P * len(ids) / self.Tp / speed

    def synth(self, text, steps=4, speed=1.0, guidance=3.0, t_shift=0.5, timings=None):
        """Return float32 48 kHz audio for `text` in the cloned voice."""
        t0 = time.perf_counter()
        if self.compute == "mps":
            wav = self.tts.generate_speech(text, self.enc, num_steps=steps, speed=speed, guidance_scale=guidance, t_shift=t_shift)
            wav = wav.squeeze().float().cpu().numpy().astype(np.float32)
            if timings is not None:
                timings.update(total_ms=(time.perf_counter() - t0) * 1000, static=False, bucket=0, frames=int(len(wav) / HOP48))
            return wav
        ids = self.tts.tokenizer.texts_to_token_ids([text])[0]
        if not ids:
            return np.zeros(0, dtype=np.float32)
        nat = self.natural_frames(ids, speed)
        use_static = self.compute == "ov" and self.buckets
        if use_static:
            N = next((b for b in self.buckets if b >= self.P + nat + 12), None)
            if N is None:   # too long for the largest bucket: split on clause boundaries and concatenate
                parts = self._split_to_fit(text, speed)
                if len(parts) > 1:
                    gap = np.zeros(int(0.12 * 48000), dtype=np.float32)
                    return np.concatenate(sum(([self.synth(p, steps, speed, guidance, t_shift), gap] for p in parts), [])[:-1])
                N = self.buckets[-1]; use_static = False   # single un-splittable chunk: fall back to CPU dynamic
        if use_static:
            gen_total = N - self.P
            # tokens that fill the whole bucket at the natural rate; fill the remainder with real filler text
            T_all = max(len(ids), int(round(gen_total * self.Tp / self.P * speed)))
            T_pad = T_all - len(ids)
            ids_all = ids + self.pad_ids[:T_pad]
            tc = self._text_condition(ids_all, N)
            t1 = time.perf_counter()
            x = self._flow(self._run_fm_static(N), tc, N, steps, guidance, t_shift)
            t2 = time.perf_counter()
            real = int(round(gen_total * len(ids) / len(ids_all))) + 3
            real = min(real, gen_total)
            feats = x[:, self.P:self.P + real]
        else:
            n = self.P + int(round(nat))
            tc = self._text_condition(ids, n)
            t1 = time.perf_counter()
            x = self._flow(self._run_fm_ort, tc, tc.shape[1], steps, guidance, t_shift)
            t2 = time.perf_counter()
            feats = x[:, self.P:]
            N = tc.shape[1]
        if torch.isnan(feats).any():
            raise RuntimeError("NaN in flow output (GPU precision?)")
        wav = self._vocode(feats)
        if use_static:
            fade = int(FADE_MS / 1000 * 48000)
            if len(wav) > fade:
                wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        t3 = time.perf_counter()
        if timings is not None:
            timings.update(text_enc_ms=(t1 - t0) * 1000, flow_ms=(t2 - t1) * 1000, voc_ms=(t3 - t2) * 1000,
                           total_ms=(t3 - t0) * 1000, bucket=N, frames=int(feats.shape[1]), static=bool(use_static))
        return wav

    def _split_to_fit(self, text, speed):
        import re
        pieces = [p.strip() for p in re.split(r"(?<=[,;:.!?])\s+", text) if p.strip()]
        out, cur = [], ""
        for p in pieces:
            cand = (cur + " " + p).strip()
            ids = self.tts.tokenizer.texts_to_token_ids([cand])[0]
            if cur and self.P + self.natural_frames(ids, speed) + 12 > self.buckets[-1]:
                out.append(cur); cur = p
            else:
                cur = cand
        if cur:
            out.append(cur)
        return out
