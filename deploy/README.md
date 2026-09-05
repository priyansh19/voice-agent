# Deployment (Mac mini)

- `com.priyansh.voice-agent.plist`: launchd user agent that runs the console/backend (`ui/server.py`) with the
  Apple Silicon overrides, at login/boot, restarting on failure. Install:
  `cp deploy/com.priyansh.voice-agent.plist ~/Library/LaunchAgents/ && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.priyansh.voice-agent.plist`
  Logs: `out/agent.log`. Restart: `launchctl kickstart -k gui/$(id -u)/com.priyansh.voice-agent`.
- The private Ollama instance (port 11436) is spawned by the agent itself; the Ollama app provides the binary.
- Bound to 0.0.0.0:8765 so it is reachable over Tailscale; put Caddy + Cloudflare Tunnel in front for the public site.
