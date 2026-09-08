# LinkedIn post

Attach, in this order: `docs/architecture_3x.png` (cover), `docs/screenshots/latency_panel_mobile.png`
(tall crop, readable on a phone), `docs/screenshots/console_scenarios.png` (the five questions),
`docs/screenshots/console_english.png`. All captured 2026-09-08 on the Core Ultra 7 155H laptop
(no discrete GPU), console at http://127.0.0.1:8765, nothing warmed by hand.

---

My laptop now answers me in a clone of my own voice, and it starts talking before I've finished the sentence.

No cloud. No API key. No NVIDIA GPU. Open-weight models only, split across the three chips an Intel Core Ultra already has: speech-to-text on the NPU, the LLM and the voice-clone decoder on the Arc iGPU, everything else on the CPU.

The interesting part is not the models. It's the clock. Every turn is measured from the moment I stop speaking (screenshots below):

⏱ transcript ready: 79 to 90 ms BEFORE I finished (it transcribes during my pause and throws the guess away if I keep talking)
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

The same code runs on a Mac mini M4 on Metal, served behind a k3s cluster with Let's Encrypt, so it doubles as a public demo.

Repo (Apache-2.0, architecture diagram, per-stage benchmarks, CI that runs without a GPU): https://github.com/priyansh19/voice-agent

I'm building in public from here on: local AI, voice, edge inference, real numbers attached. Also, I'm on the lookout for my next gig, anything in Gen AI, AI infrastructure or the messy space in between. If you're hiring for that kind of thing, or just want to nerd out about latency, my inbox is open.

Question for you: what's the longest pause you'd accept from a voice assistant before it feels broken? I've been assuming 1 second. Tell me if I'm wrong.

#VoiceAI #EdgeAI #OpenSource #BuildInPublic #GenerativeAI #AIInfrastructure #MachineLearning #LLM #MLOps #AIEngineer #ArtificialIntelligence #SoftwareEngineering #OpenToWork #Hiring #TechRecruiting #TalentAcquisition

---

## First comment (post it right after publishing)

Hardware for the numbers above: Intel Core Ultra 7 155H, 32 GB, no discrete GPU. Models: Granite Speech 5.0 (NPU), Granite 4.2 3B via Ollama and LuxTTS voice clone (Arc iGPU), Whisper large-v3-turbo for Hindi, Silero VAD and a WeSpeaker voice lock (CPU). Repo: https://github.com/priyansh19/voice-agent

## Posting tips

- Cover image first (LinkedIn uses the first image as the thumbnail), then the tall latency crop, then the scenarios screenshot, then the single-turn one.
- Tuesday to Thursday, 8 to 10 am in your audience's timezone; reply to every comment in the first hour.
- The hashtag block mixes topic tags (reach) with #OpenToWork #Hiring #TechRecruiting #TalentAcquisition (recruiter feeds). If you'd rather keep the caption clean, move the recruiter tags into the first comment; they still index.
- Put the repo link in the post and again in the first comment; LinkedIn down-ranks link-only posts less when the link is repeated in a comment.
