# Kubernetes on the Mac mini (k3s in Colima) — website + voice agent routing

The cluster hosts the website; the voice agent stays native on macOS (Metal) and is reached from the cluster via
the host's Tailscale IP. Everything here is free.

```bash
colima start --runtime docker --kubernetes --cpu 2 --memory 3 --disk 12 --vm-type vz --network-address
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
kubectl -n cert-manager rollout status deploy/cert-manager-webhook
kubectl apply -f deploy/k8s/10-site.yaml -f deploy/k8s/20-ingress.yaml -f deploy/k8s/30-cert-manager.yaml
```

Public access, two free options:

1. **Tailscale Funnel** (no router changes, TLS by Tailscale/Let's Encrypt):
   `tailscale funnel --bg --https=443 http://127.0.0.1:80` → https://priyanshs-mac-mini.tailf3c4e5.ts.net
2. **DuckDNS + cert-manager**: forward router ports 80/443 to the mini, set the DuckDNS subdomain to your public IP,
   edit HOSTNAME in `40-ingress-tls-duckdns.yaml`, apply it; cert-manager obtains and renews the certificate.

Paths: `/` website · `/agent/` voice console · `/ws`, `/audio`, `/record`, `/voice.wav` agent WebSockets/assets.
