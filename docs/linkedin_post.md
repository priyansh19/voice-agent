# LinkedIn post (copy below the line; attach docs/architecture_3x.png)

---

I built a voice assistant that answers in a clone of my own voice, runs 100% on my laptop, and starts responding in under 200 milliseconds. No cloud. No API keys. Only open-weight models.

Here is what "local" means here: an Intel Core Ultra laptop with no NVIDIA GPU. The trick was not one big model, it was splitting the work across the three compute units the chip already has:

🔹 NPU → speech-to-text (Granite Speech 5.0 TurboCTC, one pass, ~130 ms)
🔹 Arc iGPU → the LLM (Granite 4.2 3B via Ollama) and the voice-cloning TTS decoder (LuxTTS on OpenVINO)
🔹 CPU → voice activity detection, the vocoder, and a "voice lock" that only answers my voice

What makes it feel instant:
• Transcription and the LLM start during my pause, ~250 ms before the turn is even over. If I keep talking, the work is cancelled and merged into the next turn.
• A spoken acknowledgement in my cloned voice ("Hmm.", "Right.") plays at ~170 ms while the real answer is generated. First words of the answer arrive about a second after I stop.
• Text streams into speech clause by clause; the first chunk is only three words long.
• Weather questions are prefetched from the transcript before the model even runs.

It understands English, Hindi and Hinglish (Whisper large-v3-turbo kicks in only when the fast English model looks unsure), answers in the language you spoke, and reaches the internet through keyless tools: live weather, Wikipedia, web search.

Some of the bugs that cost the most time were not model problems at all: "localhost" on Windows added 1.7 s per request (IPv6 first, then fallback). Ollama's four parallel slots re-evaluated the whole prompt on every turn. A voice-cloning model produced zero frames for short sentences because of a duration formula. Measuring every stage per turn is what found all of them.

The whole thing is open source under Apache-2.0, with a test suite that runs without a GPU (the models are replaced by fakes with known latencies, and a test asserts the orchestration adds under 250 ms), CI on Linux and Windows, and a self-hosted benchmark workflow so people can share numbers from their own hardware.

Repo: https://github.com/priyansh19/voice-agent

Architecture diagram below. Happy to compare notes with anyone working on low-latency speech pipelines, especially on Apple Silicon or NVIDIA where the same design should land well under 500 ms.

#VoiceAI #OpenSource #EdgeAI #LLM #SpeechRecognition #TTS #OpenVINO #Ollama #IntelCoreUltra #LocalAI

---

Shorter variant (if you prefer a punchier post):

Built a fully local voice agent that talks back in my own cloned voice and starts responding in under 200 ms, on a laptop with no NVIDIA GPU. Open weights only: Granite Speech on the NPU, Granite 4.2 + LuxTTS on the Intel Arc iGPU, Whisper for Hindi/Hinglish, a voice lock so it only answers me, live weather/Wikipedia/search tools. Speculative execution during your pause, cloned-voice fillers, clause-level streaming. Apache-2.0, tested in CI without a GPU. https://github.com/priyansh19/voice-agent #VoiceAI #OpenSource #EdgeAI
