import ollama, time
c = ollama.Client(); M = "granite4.2:3b"
def ttft(msgs, tag):
    t = time.perf_counter(); r = c.chat(model=M, messages=msgs, stream=True, think=False, keep_alive=-1, options={"num_ctx": 4096})
    for p in r:
        if p.message.content: print(f"{tag}: TTFT {(time.perf_counter()-t)*1000:.0f}ms"); break
    return r
r = ttft([{"role": "user", "content": "Say hi in three words."}], "baseline"); [_ for _ in r]
# start a long generation, cancel after 0.4 s, then immediately ask again
r = c.chat(model=M, messages=[{"role": "user", "content": "Write a 400 word essay about the sea."}], stream=True, think=False, keep_alive=-1)
t0 = time.perf_counter(); n = 0
for p in r:
    n += 1
    if time.perf_counter() - t0 > 0.4: break
r.close(); print(f"cancelled after {n} tokens")
ttft([{"role": "user", "content": "Say hi in three words."}], "right after cancel")
