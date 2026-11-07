#!/usr/bin/env bash
# Expose the PUBLIC listener (default 127.0.0.1:8000) on the internet. Never the operator one.
#
#   scripts/ingress_up.sh [funnel|quick]
#
#   funnel (default)  Tailscale Funnel: `tailscale funnel --bg <port>`; stable https://<host>.ts.net
#   quick             Cloudflare quick tunnel: `cloudflared tunnel --url`; random trycloudflare.com URL,
#                     changes on every restart (the web client re-reads config.json, so that is fine)
#
# Idempotent: re-running reuses a live funnel/tunnel and rewrites config.json with the same URL.
# Writes the public URL into web/static/config.json and prints the SEVI_* values to set.
# Prints no secrets (none are read).
set -euo pipefail

MODE="${1:-funnel}"
PORT="${SEVI_BIND_PORT:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/web/static/config.json"
STATE_DIR="$ROOT/var/ingress"
mkdir -p "$STATE_DIR"

die() { echo "ingress_up: $*" >&2; exit 1; }

funnel_url() {
  tailscale funnel status --json 2>/dev/null | python3 -c '
import json, sys
port = sys.argv[1]
try:
    st = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for hp, on in (st.get("AllowFunnel") or {}).items():
    if not on:
        continue
    for h in ((st.get("Web") or {}).get(hp) or {}).get("Handlers", {}).values():
        if h.get("Proxy", "").rstrip("/").endswith(":" + port):
            host = hp.rsplit(":", 1)[0]
            print("https://" + host if hp.endswith(":443") else "https://" + hp)
            sys.exit(0)
' "$PORT"
}

up_funnel() {
  command -v tailscale >/dev/null || die "tailscale not installed"
  tailscale status >/dev/null 2>&1 || die "tailscale is not logged in / running"
  local url
  url="$(funnel_url || true)"
  if [[ -z "$url" ]]; then
    tailscale funnel --bg "$PORT" >/dev/null
    url="$(funnel_url || true)"
  fi
  [[ -n "$url" ]] || die "funnel started but no URL found (is Funnel enabled in the tailnet ACL?)"
  echo "$url"
}

up_quick() {
  command -v cloudflared >/dev/null || die "cloudflared not installed"
  local pidf="$STATE_DIR/cloudflared.pid" log="$STATE_DIR/cloudflared.log" url=""
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" | tail -1 || true)"
  fi
  if [[ -z "$url" ]]; then
    [[ -f "$pidf" ]] && kill "$(cat "$pidf")" 2>/dev/null || true
    : >"$log"
    nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$log" 2>&1 &
    echo $! >"$pidf"
    for _ in $(seq 1 30); do
      url="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" | tail -1 || true)"
      [[ -n "$url" ]] && break
      sleep 1
    done
  fi
  [[ -n "$url" ]] || die "no trycloudflare URL after 30 s; see $log"
  echo "$url"
}

case "$MODE" in
  funnel) URL="$(up_funnel)"; HEADER="X-Forwarded-For" ;;
  quick) URL="$(up_quick)"; HEADER="Cf-Connecting-Ip" ;;
  *) die "usage: $0 [funnel|quick]" ;;
esac

python3 - "$CONFIG" "$URL" "$MODE" <<'PY'
import json, sys
from datetime import datetime, timezone
path, url, mode = sys.argv[1:]
cfg = {"api_base": url, "ingress": mode,
       "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(cfg, f)
    f.write("\n")
import os; os.replace(tmp, path)
PY

cat <<EOF
ingress: $MODE -> 127.0.0.1:$PORT
public URL: $URL
wrote: web/static/config.json (republish Pages: make pages)

Set these for the app, then restart it (both tunnels connect from loopback):
SEVI_PUBLIC_BASE_URL=$URL
SEVI_CLIENT_IP_HEADER=$HEADER
SEVI_TRUSTED_PROXIES=127.0.0.1/32,::1/128
EOF
