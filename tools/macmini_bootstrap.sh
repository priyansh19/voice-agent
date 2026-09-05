#!/bin/bash
# Mac mini bootstrap: one-time, unattended after this. Afterwards Claude administers the machine over Tailscale SSH
# from the laptop with passwordless sudo; you never need to log in to the mini again.
#
#   TS_AUTHKEY=tskey-auth-xxxxx bash macmini_bootstrap.sh
#
# TS_AUTHKEY: a Tailscale auth key from https://login.tailscale.com/admin/settings/keys (reusable, no expiry is fine).
#             Without it the script opens the Tailscale login in a browser instead.
#
# What it does (idempotent, safe to re-run)
#   1. Xcode command-line tools, Homebrew, git, uv, node, Claude Code, Ollama, Tailscale, Caddy, cloudflared,
#      Colima + Docker CLI + kubectl (k3s for the website), ffmpeg, portaudio, jq.
#   2. Always-on: no sleep, wake on LAN, auto-restart after power loss, Remote Login (SSH) on.
#   3. Passwordless sudo for this user (so remote administration never blocks on a password).
#   4. Installs the laptop's PUBLIC SSH key (no secret is created or displayed here).
#   5. Joins Tailscale (with the auth key: unattended).
#   6. Prints the connection report to share: user, hostname, Tailscale IP. Nothing in it is secret.
set -euo pipefail

LAPTOP_PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG/fLQAFcpgC/t1Z1/7n4cH/7BNR/cLLq2Ff7/OI8Xuv claude-code@laptop-for-macmini'
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
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' "$HOME/.zprofile" 2>/dev/null || echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
brew update >/dev/null
brew install git uv node ffmpeg portaudio caddy cloudflared kubectl colima docker jq >/dev/null
brew install --cask ollama tailscale >/dev/null 2>&1 || brew install ollama tailscale >/dev/null
npm list -g @anthropic-ai/claude-code >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code >/dev/null
ok "git $(git --version | awk '{print $3}') · uv $(uv --version | awk '{print $2}') · node $(node --version) · claude $(claude --version 2>/dev/null | head -1)"

say "3/6 Always-on server settings (your Mac password once, for sudo)"
sudo -v
sudo pmset -a sleep 0 disksleep 0 displaysleep 10 womp 1 autorestart 1 >/dev/null
sudo systemsetup -setremotelogin on >/dev/null 2>&1 || true
ok "no sleep · wake on LAN · auto-restart · Remote Login on"

say "4/6 Passwordless sudo for $USER (remote administration must never block on a password)"
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/90-voice-agent-admin >/dev/null
sudo chmod 440 /etc/sudoers.d/90-voice-agent-admin
ok "configured"

say "5/6 SSH access for the laptop (public key only) + Tailscale"
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys" && chmod 600 "$HOME/.ssh/authorized_keys"
grep -qF "$LAPTOP_PUBKEY" "$HOME/.ssh/authorized_keys" || echo "$LAPTOP_PUBKEY" >> "$HOME/.ssh/authorized_keys"
ok "laptop key authorized"
open -a Tailscale 2>/dev/null || true
sleep 3
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
[[ -x "$TS" ]] || TS=$(command -v tailscale)
if [[ -n "${TS_AUTHKEY:-}" ]]; then
  "$TS" up --authkey="$TS_AUTHKEY" --ssh --hostname=macmini-voice >/dev/null 2>&1 && ok "joined tailnet unattended (hostname macmini-voice)"
else
  echo "   No TS_AUTHKEY given: sign in to Tailscale in the window that opened (once)."
  "$TS" up --ssh --hostname=macmini-voice 2>/dev/null || true
fi

say "6/6 Ollama"
open -a Ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &)
sleep 3
ollama pull granite4.2:3b >/dev/null 2>&1 && ok "granite4.2:3b pulled" || echo "   (pull failed, Claude will retry over SSH)"

# ---------------------------------------------------------------- report
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "n/a")
TS_IP=$("$TS" ip -4 2>/dev/null | head -1 || echo "not joined")
TS_NAME=$("$TS" status --json 2>/dev/null | jq -r '.Self.DNSName' 2>/dev/null | sed 's/\.$//' || echo "n/a")
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
  echo
  echo "Share the lines above with Claude (none are secrets). Nothing else to do on the mini."
} | tee "$REPORT"
say "Done. Report saved to $REPORT"
