"""Granite 4.2 3B via Ollama (Vulkan on the Intel Arc iGPU): streaming chat with native tool calling.
Each turn uses its own HTTP client so a cancelled turn can be aborted from another thread by closing the socket
(Ollama then stops generating immediately and the next request is not queued behind it)."""
import threading, time
import ollama
from .tools import TOOLS, run_tool


class OllamaLLM:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.options = {"num_ctx": cfg.num_ctx, "temperature": cfg.temperature, "num_predict": cfg.num_predict}

    def _client(self):
        return ollama.Client(host=self.cfg.host)

    def warmup(self):
        """Load the model on the GPU and prime the KV cache with the system prompt (+ tool schema)."""
        t = time.perf_counter()
        r = self._client().chat(model=self.cfg.model, messages=[{"role": "system", "content": self.cfg.system_prompt},
                                                                {"role": "user", "content": "hi"}], tools=TOOLS,
                                think=self.cfg.think, keep_alive=self.cfg.keep_alive, options=dict(self.options, num_predict=1))
        self.log(f"[llm] warm ({(time.perf_counter()-t)*1000:.0f}ms, prompt {r.prompt_eval_count} tok)")

    def stream(self, history, user_text, cancel: threading.Event, on_meta=None, on_client=None, on_tool=None, max_rounds=4):
        """Yield text deltas. Runs the tool loop transparently.
        on_client(client): lets the caller abort the socket.  on_tool(name): called when a tool call starts."""
        msgs = [{"role": "system", "content": self.cfg.system_prompt}] + list(history) + [{"role": "user", "content": user_text}]
        for _ in range(max_rounds):
            client = self._client()
            if on_client: on_client(client)
            content, tool_calls = "", []
            try:
                resp = client.chat(model=self.cfg.model, messages=msgs, tools=TOOLS, stream=True,
                                   think=self.cfg.think, keep_alive=self.cfg.keep_alive, options=self.options)
                for part in resp:
                    if cancel.is_set():
                        return
                    m = part.message
                    if m.content:
                        content += m.content
                        yield m.content
                    if m.tool_calls:
                        tool_calls.extend(m.tool_calls)
                    if part.done and on_meta:
                        on_meta(part)
            except Exception as e:
                if cancel.is_set():
                    return
                self.log(f"[llm] error: {e!r}"); return
            finally:
                try: client._client.close()
                except Exception: pass
            if not tool_calls or cancel.is_set():
                return
            msgs.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc in tool_calls:
                name, args = tc.function.name, tc.function.arguments or {}
                if on_tool: on_tool(name)
                result = run_tool(name, args, self.log)
                msgs.append({"role": "tool", "content": result, "tool_name": name})

    @staticmethod
    def abort(client):
        """Close the socket of an in-flight request from another thread."""
        try: client._client.close()
        except Exception: pass
