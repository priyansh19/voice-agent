"""Reproduce the agent's exact LLM call and measure time-to-first-token across consecutive turns."""
import os, sys
ROOT = next(p for p in [os.path.dirname(os.path.abspath(__file__))] + [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * i))) for i in range(1, 5)] if os.path.exists(os.path.join(p, "pyproject.toml")))
import sys, time, ollama
sys.path.insert(0, ROOT)
from agent import config as C
from agent.llm import TOOLS
cfg = C.load(); M = cfg.llm.model
opts = {"num_ctx": cfg.llm.num_ctx, "temperature": cfg.llm.temperature, "num_predict": cfg.llm.num_predict}
def turn(msgs, tools, tag):
    c = ollama.Client(host=cfg.llm.host)
    t = time.perf_counter(); first = None; last = None
    for p in c.chat(model=M, messages=msgs, tools=tools, stream=True, think=False, keep_alive=-1, options=opts):
        if first is None and p.message.content: first = time.perf_counter() - t
        last = p
    print(f"{tag:34s} TTFT {first*1000 if first else -1:6.0f}ms | prompt {last.prompt_eval_count} tok {last.prompt_eval_duration/1e6:.0f}ms | load {last.load_duration/1e6:.0f}ms | total {last.total_duration/1e6:.0f}ms | gen {last.eval_count} tok {last.eval_duration/1e6:.0f}ms")
    c._client.close()
sysm = {"role": "system", "content": cfg.llm.system_prompt}
h = [sysm, {"role": "user", "content": "What is the capital of Italy?"}]
turn(h, TOOLS, "1 tools, fresh")
turn(h, TOOLS, "2 tools, identical (cache?)")
h2 = h + [{"role": "assistant", "content": "Rome is the capital of Italy."}, {"role": "user", "content": "And what is it famous for?"}]
turn(h2, TOOLS, "3 tools, history +1 turn")
turn(h2, None, "4 NO tools, same history")
turn(h2, None, "5 NO tools, repeat")
turn(h2, TOOLS, "6 tools again")
