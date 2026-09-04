"""Granite Speech 5.0 470M TurboCTC (Apache-2.0) on OpenVINO with static-shape buckets.

On an Intel Core Ultra the NPU transcribes a 6 s bucket in ~120-150 ms and is otherwise idle,
leaving the GPU to the LLM and TTS.  Encoder-only CTC: one forward pass, greedy decode, no autoregression.
"""
import os, time
import numpy as np
from .config import abspath

FPS = 50  # encoder input frames per second (20 ms hop)


class GraniteSTT:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.model_id = cfg.model_id
        self.device = cfg.device
        self.buckets = sorted(cfg.buckets_frames)
        self.ov_dir = abspath(cfg.ov_dir)
        os.makedirs(self.ov_dir, exist_ok=True)
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.model_id)
        import openvino as ov
        self.core = ov.Core()
        self.core.set_property({"CACHE_DIR": os.path.join(self.ov_dir, "cache")})
        import threading
        self.lock = threading.Lock()     # one infer request per bucket: serialize overlapping speculative turns
        self.reqs = {}
        for N in self.buckets:
            xml = os.path.join(self.ov_dir, f"model_static{N}.xml")
            if not os.path.exists(xml):
                self._export(N, xml)
            t = time.perf_counter()
            cm = self.core.compile_model(xml, self.device, {"PERFORMANCE_HINT": "LATENCY"})
            self.reqs[N] = cm.create_infer_request()
            self.log(f"[stt] bucket {N} frames ({N/FPS:.0f}s) compiled on {self.device} in {time.perf_counter()-t:.1f}s")

    def _export(self, N, xml):
        import torch, openvino as ov
        from transformers import AutoModelForCTC
        self.log(f"[stt] exporting static OpenVINO model for {N} frames (one-time)...")
        model = AutoModelForCTC.from_pretrained(self.model_id, dtype=torch.float32).eval()
        a = np.random.randn(int(N / FPS * 16000)).astype(np.float32) * 0.01
        ex = self.proc([a], sampling_rate=16000)

        class Wrap(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, input_features, attention_mask):
                return s.m(input_features=input_features, attention_mask=attention_mask).logits
        ovm = ov.convert_model(Wrap(model), example_input=dict(ex),
                               input=[("input_features", [1, ex["input_features"].shape[1], 320]),
                                      ("attention_mask", [1, ex["attention_mask"].shape[1]])])
        ov.save_model(ovm, xml)

    def warmup(self):
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio16k: np.ndarray) -> str:
        """audio16k: float32 mono at 16 kHz. Returns lower-case text without punctuation."""
        n = len(audio16k)
        N = next((b for b in self.buckets if n <= b / FPS * 16000), self.buckets[-1])
        cap = int(N / FPS * 16000)
        if n > cap:                      # longer than the largest bucket: keep the most recent part
            audio16k = audio16k[-cap:]; n = cap
        buf = np.zeros(cap, dtype=np.float32); buf[:n] = audio16k
        inp = self.proc([buf], sampling_rate=16000)
        mask = np.zeros(inp["attention_mask"].shape, dtype=np.int64)
        mask[:, : min(mask.shape[1], int(np.ceil(n / 16000 * FPS)) + 1)] = 1
        req = self.reqs[N]
        with self.lock:
            req.infer({"input_features": inp["input_features"].numpy(), "attention_mask": mask})
            logits = np.array(req.get_output_tensor(0).data)
        ids = logits.argmax(-1)
        import torch
        return self.proc.batch_decode(torch.from_numpy(ids), skip_special_tokens=True)[0].strip()
