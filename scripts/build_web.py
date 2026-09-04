"""Build the static, hostable copy of the console (web/) for Vercel / any static host.
The page talks to the backend given by ?backend=... or the field in the header (saved in localStorage)."""
import os, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
html = open(os.path.join(ROOT, "ui", "index.html"), encoding="utf-8").read()
banner = ('<div style="background:#1d2a40;color:#c9d4ea;padding:8px 20px;font-size:13px">Hosted console. '
          'Enter your backend URL (e.g. the Cloudflare tunnel of the machine running <code>ui/server.py</code>) in the '
          '<b>backend</b> field, then tick <b>Use THIS browser\'s microphone</b>.</div>')
html = html.replace("<header>", banner + "<header>", 1)
open(os.path.join(ROOT, "web", "index.html"), "w", encoding="utf-8").write(html)
open(os.path.join(ROOT, "web", "vercel.json"), "w", encoding="utf-8").write('{ "cleanUrls": true, "trailingSlash": false }\n')
print("wrote web/index.html and web/vercel.json")
