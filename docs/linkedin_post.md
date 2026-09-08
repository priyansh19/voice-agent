# LinkedIn post

Attach, in this order: `docs/architecture_3x.png`, `docs/screenshots/console_english.png`, `docs/screenshots/console_hindi.png`
(LinkedIn shows the first image as the cover, so the diagram goes first). The screenshots are real turns
captured on 2026-09-08 on the Core Ultra 7 155H laptop, console at http://127.0.0.1:8765.

---

I built a voice assistant that talks back in a clone of MY voice. It runs entirely on my laptop. And no, my laptop does not have an NVIDIA GPU.

It starts responding in about 170 ms. Faster than my colleagues on a Monday.

Backstory: every "AI voice agent" demo I saw had three things in common. A cloud API key. A monthly bill. And a pause after every question long enough to make eye contact with the audience.

So I set myself a stubborn rule: open-weight models only, zero cloud, zero API keys, and it has to answer before the silence gets awkward.

What I got on an Intel Core Ultra laptop (no discrete GPU, just the NPU, the tiny Arc iGPU and the CPU, all three working overtime):

🎙️ Hears my voice, ignores everyone else's. A "voice lock" (WeSpeaker CAM++) compares every utterance to my recording. My roommate is officially not authorised to ask it for the weather.

🧠 Transcribes on the NPU (Granite Speech 5.0 TurboCTC) in ~100 ms. Yes, the NPU that ships in these laptops for "background blur" is doing real work.

💬 Thinks with Granite 4.2 3B on the iGPU, streams the answer clause by clause, and starts talking after the first three words.

🗣️ Speaks in my cloned voice (LuxTTS, OpenVINO). My mother has already been fooled once. Sorry, Mummy.

🌍 Understands English, Hindi and Hinglish, and replies in whatever you spoke. Whisper large-v3-turbo wakes up only when the fast English model looks confused.

⚡ The latency trick: it starts transcribing and thinking during my PAUSE, about 250 ms before I actually stop. If I keep talking, it throws the guess away and merges. And while the real answer is being generated, it says "Hmm." in my own voice at ~170 ms. Humans do this. Now my laptop does too.

🌐 Live weather, Wikipedia and web search through keyless tools. Weather is prefetched from the transcript before the LLM even runs.

The bugs that hurt the most were not AI problems at all:
• "localhost" on Windows cost 1.7 seconds per request. IPv6 tries first, then gives up. Use 127.0.0.1. I lost a weekend to nine characters.
• Ollama's four parallel slots re-read the whole prompt every turn. One private instance, one slot, problem gone.
• The voice model produced exactly zero audio for short sentences because of a duration formula. Silence is a very hard bug to hear.

Real numbers from the console today (screenshots below), English question, nothing warmed up by hand:
• transcript ready 82 ms BEFORE I finished the sentence
• "Hmm." in my voice at 169 ms
• first LLM token at 328 ms
• first words of the actual answer at 909 ms

Hindi is honest too: about 5 s, because Whisper large-v3-turbo does the heavy lifting there. Next milestone is fixing exactly that. The same code on a Mac mini M4 runs on Metal and lands in the same range.

Everything is open source under Apache-2.0: architecture diagram, benchmarks for each stage, a test suite that runs without a GPU (the models are swapped for fakes with known latencies, and a test fails if the orchestration itself adds more than 250 ms), CI on Linux and Windows.

👉 https://github.com/priyansh19/voice-agent

This is post 1 of a series. I'm building things in public now: local AI, voice, edge inference, and the occasional self-inflicted bug. Follow along if you like your engineering with real numbers attached.

And if your team is building anything in speech, on-device inference or real-time systems, my DMs are open. I'd love to hear what you're working on.

#VoiceAI #OpenSource #EdgeAI #LocalAI #LLM #SpeechRecognition #TextToSpeech #OpenVINO #Ollama #IntelCoreUltra #BuildInPublic #MachineLearning

---

## Short variant (comment or repost text)

My laptop now answers me in my own voice in under 200 ms, with no GPU, no cloud and no API key. Open weights only: Granite Speech on the NPU, Granite 4.2 + LuxTTS on the Intel iGPU, Whisper for Hindi/Hinglish, a voice lock so it ignores everyone but me. The secret is starting to think during your pause and saying "Hmm." while it works. Apache-2.0, tested in CI without a GPU. https://github.com/priyansh19/voice-agent #VoiceAI #OpenSource #EdgeAI

## Posting tips

- Post the long version as the caption; add the diagram first, then the two or three console screenshots.
- First comment: the repo link again plus one line on hardware ("runs on a Core Ultra 7 155H, 32 GB, no dGPU").
- Best times: Tuesday to Thursday, 8 to 10 am in your audience's timezone.
- Reply to every comment in the first hour; LinkedIn rewards early conversation.
