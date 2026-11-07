#!/usr/bin/env bash
# Expose the PUBLIC listener (127.0.0.1:${SEVI_BIND_PORT:-8000}) on the internet (B spec §12.4).
# Never the operator listener.
#
#   scripts/ingress_up.sh [funnel|quick]
#
#   funnel (default)  Tailscale Funnel: `tailscale funnel --bg <port>`; stable https://<host>.ts.net
#   quick             Cloudflare quick tunnel: `cloudflared tunnel --url`; random trycloudflare.com URL
#
# Each mode tears the other one down first: there is never more than one public entry point.
# Writes var/run/ingress.env (sourced by `make serve`). Rewrites web/static/config.json only when
# the URL or mode changed. Safe to re-run. Reads and prints no secrets.
set -euo pipefail

MODE="${1:-funnel}"
PORT="${SEVI_BIND_PORT:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/web/static/config.json"
RUN="$ROOT/var/run"
PIDF="$RUN/cloudflared.pid"
LOG="$RUN/cloudflared.log"
TRYCF='https://[a-z0-9-]+\.trycloudflare\.com'
mkdir -p "$RUN"

die() { echo "ingress_up: $*" >&2; exit 1; }

stop_quick() {
  if [[ -f "$PIDF" ]]; then
    kill "$(cat "$PIDF")" 2>/dev/null || true
    rm -f "$PIDF"
  fi
  pkill -f "cloudflared tunnel --no-autoupdate" 2>/dev/null || true
}

stop_funnel() {
  if command -v tailscale >/dev/null; then
    tailscale funnel reset >/dev/null 2>&1 || true
  fi
}

up_funnel() {
  command -v tailscale >/dev/null || die "tailscale not installed"
  tailscale status >/dev/null 2>&1 || die "tailscale is not logged in / running"
  stop_quick
  tailscale funnel --bg "$PORT" >/dev/null
  tailscale serve status --json >"$RUN/serve-config.json" 2>/dev/null || true
  local host
  host="$(tailscale status --json | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
  [[ -n "$host" ]] || die "no MagicDNS name (enable MagicDNS + HTTPS certificates)"
  echo "https://$host"
}

up_quick() {
  command -v cloudflared >/dev/null || die "cloudflared not installed"
  stop_funnel
  local url=""
  if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    url="$(grep -Eo "$TRYCF" "$LOG" | tail -1 || true)"  # reuse the live tunnel
  fi
  if [[ -z "$url" ]]; then
    stop_quick
    nohup cloudflared tunnel --no-autoupdate --loglevel warn --url "http://127.0.0.1:$PORT" \
      >"$LOG" 2>&1 &
    echo $! >"$PIDF"
    for _ in $(seq 1 60); do
      url="$(grep -Eo "$TRYCF" "$LOG" | tail -1 || true)"
      [[ -n "$url" ]] && break
      sleep 1
    done
  fi
  [[ -n "$url" ]] || die "quick tunnel did not come up in 60 s; see var/run/cloudflared.log"
  echo "$url"
}

case "$MODE" in
  funnel)
    URL="$(up_funnel)"
    HDR="$(cat "$ROOT/ops/funnel_client_ip_header.txt" 2>/dev/null || echo "X-Forwarded-For")"
    ;;
  quick)
    URL="$(up_quick)"
    HDR="CF-Connecting-IP"
    ;;
  *) die "usage: $0 [funnel|quick]" ;;
esac

# Atomic writes (tmp + rename).
ENV_TMP="$RUN/ingress.env.tmp"
printf 'SEVI_PUBLIC_BASE_URL=%s\nSEVI_CLIENT_IP_HEADER=%s\nSEVI_TRUSTED_PROXIES=%s\n' \
  "$URL" "$HDR" "127.0.0.1/32,::1/128" >"$ENV_TMP"
mv "$ENV_TMP" "$RUN/ingress.env"

CHANGED="$(python3 - "$CONFIG" "$URL" "$MODE" <<'PY'
import json, os, sys
from datetime import datetime, timezone
path, url, mode = sys.argv[1:]
try:
    with open(path) as f:
        old = json.load(f)
except (OSError, ValueError):
    old = {}
if old.get("api_base") == url and old.get("ingress") == mode:
    print("no")
    sys.exit(0)
cfg = {"api_base": url, "ingress": mode,
       "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(path + ".tmp", "w") as f:
    json.dump(cfg, f)
    f.write("\n")
os.replace(path + ".tmp", path)
print("yes")
PY
)"

echo "ingress: $MODE -> 127.0.0.1:$PORT"
echo "public URL: $URL"
echo "wrote: var/run/ingress.env (make serve loads it; restart serve if the mode changed)"
cat "$RUN/ingress.env"
if [[ "$CHANGED" == "yes" ]]; then
  echo "api_base changed: web/static/config.json updated -> republish the site (make pages)"
fi
curl -fsS --max-time 10 "$URL/healthz" >/dev/null \
  || echo "warning: $URL/healthz not reachable yet (is sevi serve running?)" >&2
