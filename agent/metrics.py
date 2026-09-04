"""Per-turn latency breakdown. Times are perf_counter seconds; 'speech_end' is the firm end-of-turn."""
from dataclasses import dataclass, field
from rich.table import Table
from rich.console import Console

console = Console()


@dataclass
class TurnMetrics:
    turn_id: int
    tentative_end: float = 0.0
    speech_end: float = 0.0
    stt_done: float = 0.0
    llm_first_token: float = 0.0
    first_chunk_text: float = 0.0
    first_audio_ready: float = 0.0
    first_audio_played: float = 0.0
    filler_played: float = 0.0
    llm_done: float = 0.0
    audio_done: float = 0.0
    transcript: str = ""
    response: str = ""
    speculative: bool = False
    cancelled: bool = False
    llm_prompt_tokens: int = 0
    llm_eval_tokens: int = 0
    llm_prompt_ms: float = 0.0
    llm_eval_ms: float = 0.0
    tts_chunks: list = field(default_factory=list)

    def ms(self, a, b):
        return (b - a) * 1000 if a and b else float("nan")

    def summary(self):
        return {"first_audio_ms": self.ms(self.speech_end, self.first_audio_played),
                "filler_ms": self.ms(self.speech_end, self.filler_played),
                "stt_ms": self.ms(self.speech_end, self.stt_done),
                "llm_first_token_ms": self.ms(self.speech_end, self.llm_first_token)}

    def rows(self):
        se = self.speech_end
        tps = 1000 * self.llm_eval_tokens / self.llm_eval_ms if self.llm_eval_ms else 0
        rows = [
            ("speculative head start", -self.ms(self.tentative_end, se), "STT+LLM started this many ms before the firm end"),
            ("STT done", self.ms(se, self.stt_done), f"'{self.transcript[:70]}'"),
            ("LLM first token", self.ms(se, self.llm_first_token), f"prompt {self.llm_prompt_tokens} tok in {self.llm_prompt_ms:.0f} ms"),
            ("first TTS chunk text", self.ms(se, self.first_chunk_text), ""),
            ("filler audio out", self.ms(se, self.filler_played), "cloned-voice filler"),
            ("first audio ready", self.ms(se, self.first_audio_ready), self.tts_chunks[0] if self.tts_chunks else ""),
            ("FIRST ANSWER AUDIO OUT", self.ms(se, self.first_audio_played), "<-- response latency"),
            ("LLM done", self.ms(se, self.llm_done), f"{self.llm_eval_tokens} tok @ {tps:.1f} tok/s"),
            ("audio done", self.ms(se, self.audio_done), f"{len(self.tts_chunks)} chunks"),
        ]
        return [(name, round(v), note) for name, v, note in rows if v == v]

    def report(self):
        t = Table(title=f"Turn {self.turn_id}: ms after the firm end of your speech")
        t.add_column("stage"); t.add_column("ms", justify="right"); t.add_column("note")
        for name, v, note in self.rows():
            t.add_row(name, f"{v}", note)
        console.print(t)
