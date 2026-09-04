import sys, time, json, httpx
sys.path.insert(0, ".")
from agent import config as C
from agent.llm import TOOLS
cfg = C.load(); M = cfg.llm.model
def run(tag, body):
    body = dict(model=M, stream=True, keep_alive=-1, options={"num_ctx": 4096, "num_predict": 40}, **body)
    t = time.perf_counter(); arrivals = []; content_first = None; n = 0
    with httpx.Client(timeout=60) as c, c.stream("POST", f"{cfg.llm.host}/api/chat", json=body) as r:
        for line in r.iter_lines():
            if not line: continue
            d = json.loads(line); n += 1; dt = (time.perf_counter() - t) * 1000
            if len(arrivals) < 3: arrivals.append(f"{dt:.0f}ms:{d['message'].get('content','')[:12]!r}")
            if content_first is None and d["message"].get("content"): content_first = dt
            if d.get("done"): total = d["total_duration"] / 1e6; pe = d["prompt_eval_duration"] / 1e6
    print(f"{tag:40s} first content {(content_first or -1):6.0f}ms | chunks {n} | server total {total:.0f}ms prompt {pe:.0f}ms | first chunks {arrivals}")
u = [{"role": "user", "content": "Name three colors."}]
s = [{"role": "system", "content": cfg.llm.system_prompt}] + u
for i in range(3): run(f"B{i} plain think=False", dict(messages=u, think=False))
for i in range(3): run(f"D{i} system think=False", dict(messages=s, think=False))
run("B again after D", dict(messages=u, think=False))
run("F system+tools think=False", dict(messages=s, tools=TOOLS, think=False))
run("F again", dict(messages=s, tools=TOOLS, think=False))
