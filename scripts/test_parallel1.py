"""TTFT with OLLAMA_NUM_PARALLEL=1 (second instance on :11435) vs the default instance (:11434)."""
import sys, time, json, httpx
sys.path.insert(0, ".")
from agent import config as C
from agent.tools import TOOLS
cfg = C.load()
def turn(host, msgs, tag):
    body = dict(model=cfg.llm.model, stream=True, keep_alive=-1, think=False, tools=TOOLS, messages=msgs,
                options={"num_ctx": cfg.llm.num_ctx, "temperature": 0.6, "num_predict": 30})
    t = time.perf_counter(); first = None; last = None
    with httpx.Client(timeout=120) as c, c.stream("POST", f"{host}/api/chat", json=body) as r:
        for line in r.iter_lines():
            if not line: continue
            d = json.loads(line)
            if first is None and d["message"].get("content"): first = time.perf_counter() - t
            if d.get("done"): last = d
    print(f"  {tag:28s} TTFT {first*1000:5.0f}ms | prompt {last['prompt_eval_count']:4d} tok in {last['prompt_eval_duration']/1e6:4.0f}ms")
sysm = {"role": "system", "content": cfg.llm.system_prompt}
hist = [sysm, {"role": "user", "content": "[user spoke English; reply in English] What is the capital of Italy?"}]
for host in ["http://127.0.0.1:11434", "http://127.0.0.1:11435"]:
    print(host)
    try:
        turn(host, hist, "warm")
        h = list(hist)
        for i, q in enumerate(["And what is it famous for?", "How far is it from Milan?", "Is it worth visiting in winter?"]):
            h = h + [{"role": "assistant", "content": "Rome."}, {"role": "user", "content": f"[user spoke English; reply in English] {q}"}]
            turn(host, h, f"turn {i+2} (history grows)")
    except Exception as e:
        print("  FAILED", e.__class__.__name__, str(e)[:100])
