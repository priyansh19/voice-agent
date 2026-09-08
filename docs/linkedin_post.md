# LinkedIn post

Attach, in this order: `docs/architecture_3x.png` (cover), `docs/screenshots/console_scenarios.png`,
`docs/screenshots/console_english.png`. Both screenshots are real turns captured on 2026-09-08 on the
Core Ultra 7 155H laptop (no discrete GPU), console at http://127.0.0.1:8765, nothing warmed by hand.

---

My laptop now answers me in a clone of my own voice, and it starts talking before I've finished the sentence.

No cloud. No API key. No NVIDIA GPU. Open-weight models only, split across the three chips an Intel Core Ultra already has: speech-to-text on the NPU, the LLM and the voice-clone decoder on the Arc iGPU, everything else on the CPU.

The interesting part is not the models. It's the clock. Every turn is measured from the moment I stop speaking (screenshots below):

⏱ transcript ready: 79 to 90 ms BEFORE I finished (it starts transcribing during my pause and throws the guess away if I keep talking)
⏱ "Hmm." in my own cloned voice: 166 to 174 ms
⏱ first LLM token: 189 to 477 ms
⏱ first words of the actual answer: 786 to 994 ms

Same numbers, different questions, because latency is only worth quoting on random questions, not the one you tuned for:
• "Explain the CAP theorem in one sentence" → 786 ms
• "What happens in the first 100 ms after I press Enter on a URL" → 868 ms
• "Reverse a singly linked list in place" → 989 ms
• "One tough question for a senior backend candidate" → 994 ms
• "A haiku about a Kubernetes pod that keeps restarting" → 1.8 s (the model paused to think; the haiku was worth it)

It also understands Hindi and Hinglish and replies in the language you spoke, but those turns take 4 to 5 s today. That's the next milestone, and I'll post the numbers when it moves.

Repo (Apache-2.0, architecture diagram, per-stage benchmarks, CI that runs without a GPU): https://github.com/priyansh19/voice-agent

Building in public from here on: local AI, voice, edge inference, real numbers attached. If your team works on speech, on-device inference or real-time systems, my DMs are open.

#VoiceAI #OpenSource #EdgeAI #LocalAI #LLM #SpeechRecognition #TextToSpeech #OpenVINO #Ollama #IntelCoreUltra #BuildInPublic

---

## First comment (post it right after publishing)

Hardware for the numbers above: Intel Core Ultra 7 155H, 32 GB, no discrete GPU. Models: Granite Speech 5.0 (NPU), Granite 4.2 3B via Ollama and LuxTTS voice clone (Arc iGPU), Whisper large-v3-turbo for Hindi, Silero VAD and a WeSpeaker voice lock (CPU). Same code runs on a Mac mini M4 with Metal. Repo: https://github.com/priyansh19/voice-agent

## Posting tips

- Cover image first (LinkedIn uses the first image as the thumbnail), then the scenarios screenshot, then the single-turn one.
- Tuesday to Thursday, 8 to 10 am in your audience's timezone; reply to every comment in the first hour.
