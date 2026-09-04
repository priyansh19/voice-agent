# Security and responsible use

**Reporting**: open a private security advisory on GitHub (Security → Advisories → Report a vulnerability) rather
than a public issue. You will get a response within a week.

**Voice cloning**: this project clones the voice of whoever records the reference. Only clone your own voice or
a voice you have explicit permission to clone, and do not expose a cloned-voice agent publicly without access
control (see README → Hosting: put Cloudflare Access or an equivalent in front of a tunnel).

**Data**: everything runs locally; the only outbound traffic is the keyless tool calls (Open-Meteo, Wikipedia,
DuckDuckGo) and model downloads from Hugging Face / GitHub releases on first run. Your reference recording and
voiceprint stay in `voices/` and `models/voice_cache/`, both git-ignored.

**Tool execution**: `calculate` evaluates arithmetic only (character allow-list, no builtins); no tool writes to
disk or runs commands. Keep it that way in contributions, or gate new tools behind explicit user confirmation.
