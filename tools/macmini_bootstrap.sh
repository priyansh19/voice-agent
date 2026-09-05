#!/bin/bash
# Mac mini bootstrap for the voice-agent server.
# Run once on the mini:  bash macmini_bootstrap.sh
# Idempotent: safe to re-run. Needs your Mac password for sudo steps (typed on the mini, never shared).
#
# What it does
#   1. Xcode command-line tools, Homebrew, git, uv, node, Claude Code, Ollama, Tailscale, Caddy, cloudflared,
#      Colima + Docker CLI + kubectl (k3s inside Colima for the website), ffmpeg, portaudio.
#   2. Keeps the mini awake and reachable: no sleep, wake on LAN, Remote Login (SSH) on.
#   3. Installs the laptop's PUBLIC SSH key so the laptop can log in without passwords (nothing secret is created
#      or shown here).
#   4. Clones the repo, creates the two Python environments, pulls the LLM.
#   5. Prints a report of everything I need to connect: user, hostname, LAN IP, Tailscale IP. Share THAT.
set -euo pipefail

LAPTOP_PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG/fLQAFcpgC/t1Z1/7n4cH/7BNR/cLLq2Ff7/OI8Xuv claude-code@laptop-for-macmini'
REPO_URL='https://github.com/priyansh19/voice-agent.git'
REPO_DIR="$HOME/voice-agent"
REPORT="$HOME/macmini-setup-report.txt"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m   ✓ %s\033[0m\n' "$*"; }

[[ "$(uname)" == "Darwin" ]] || { echo "This script is for macOS."; exit 1; }
[[ "$(uname -m)" == "arm64" ]] || echo "warning: not Apple Silicon; the Metal/MPS paths will not apply"

say "1/6 Xcode command-line tools"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  echo "   A dialog opened: click Install, wait for it to finish, then re-run this script."
  exit 0
fi
ok "present"

say "2/6 Homebrew and packages"
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
brew update >/dev/null
brew install git uv node gh ffmpeg portaudio caddy cloudflared kubectl colima docker jq >/dev/null
brew install --cask ollama tailscale >/dev/null 2>&1 || brew install ollama tailscale >/dev/null
ok "$(git --version | head -1) · uv $(uv --version | cut -d' ' -f2) · node $(node --version)"

say "3/6 Claude Code"
npm list -g @anthropic-ai/claude-code >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code >/dev/null
ok "claude $(claude --version 2>/dev/null | head -1)"

say "4/6 Always-on server settings (sudo)"
sudo pmset -a sleep 0 disksleep 0 displaysleep 10 womp 1 autorestart 1 >/dev/null
sudo systemsetup -setremotelogin on >/dev/null 2>&1 || true
ok "no sleep, auto-restart after power loss, Remote Login (SSH) on"

say "5/6 SSH access for the laptop (public key only)"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys" && chmod 600 "$HOME/.ssh/authorized_keys"
grep -qF "$LAPTOP_PUBKEY" "$HOME/.ssh/authorized_keys" || echo "$LAPTOP_PUBKEY" >> "$HOME/.ssh/authorized_keys"
ok "laptop key authorized"
if ! tailscale status >/dev/null 2>&1; then
  echo "   Tailscale: opening login (sign in with the same account you will use on the laptop)."
  open -a Tailscale || true
  sudo tailscale up 2>/dev/null || true
fi

say "6/6 Repository, environments, models"
if [[ -d "$REPO_DIR/.git" ]]; then git -C "$REPO_DIR" pull -q || true; else
  gh auth status >/dev/null 2>&1 || { echo "   GitHub login needed for the private repo (browser opens):"; gh auth login -h github.com -p https -w; }
  gh repo clone "$REPO_URL" "$REPO_DIR" -- -q
fi
cd "$REPO_DIR"
uv python install 3.12 >/dev/null
uv sync --group dev >/dev/null
[[ -d tts_worker/.venv ]] || uv venv --python 3.12 tts_worker/.venv >/dev/null
uv pip install --python tts_worker/.venv -r tts_worker/requirements.txt >/dev/null 2>&1 || echo "   (tts_worker env: will be finished during the Apple Silicon port)"
ok "python environments"
open -a Ollama 2>/dev/null || (ollama serve >/dev/null 2>&1 &)
sleep 3
ollama pull granite4.2:3b >/dev/null 2>&1 && ok "granite4.2:3b pulled" || echo "   (ollama pull failed; will retry later)"

# ---------------------------------------------------------------- report
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "n/a")
TS_IP=$(tailscale ip -4 2>/dev/null | head -1 || echo "not logged in")
TS_NAME=$(tailscale status --json 2>/dev/null | jq -r '.Self.DNSName' 2>/dev/null | sed 's/\.$//' || echo "n/a")
{
  echo "Mac mini setup report — $(date)"
  echo "user:            $USER"
  echo "hostname:        $(scutil --get LocalHostName).local"
  echo "lan ip:          $LAN_IP"
  echo "tailscale ip:    $TS_IP"
  echo "tailscale name:  $TS_NAME"
  echo "chip / memory:   $(sysctl -n machdep.cpu.brand_string) / $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  echo "macos:           $(sw_vers -productVersion)"
  echo "host key fp:     $(ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub 2>/dev/null | awk '{print $2}')"
  echo "repo:            $REPO_DIR"
  echo "ollama:          $(ollama --version 2>/dev/null | head -1)"
  echo
  echo "Share the lines above with Claude (none of them are secrets)."
  echo "Test from the laptop:  ssh -i ~/.ssh/macmini $USER@$TS_IP uname -a"
} | tee "$REPORT"
say "Done. Report saved to $REPORT"
