#!/usr/bin/env bash
# Demo day, one command (browser only; phone stays off).
#
#   scripts/demo_up.sh up     Evon on Kaggle (if not already up) -> Funnel ingress -> sevi serve
#                             in the background with the live config -> wait for readiness L0.
#   scripts/demo_up.sh down   stop sevi serve, reset the Funnel, stop the Kaggle Evon session.
#   scripts/demo_up.sh status readiness summary.
#
# Secrets come from Infisical (env "dev": Gnani keys, Evon key, allowlist keys, operator password);
# nothing secret is printed. DEMO_BUDGET (default 2500) caps credits per key for the session;
# DEMO_STOP_SESSIONS (default 3) is the voice-session buffer held back on the public key. The non-secret live config is below and can be overridden from env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN="$ROOT/var/run"
PIDF="$RUN/serve.pid"
LOG="$RUN/serve.log"
PROJECT="${INFISICAL_PROJECT_ID:-595a2714-fcb6-4450-aa0d-60de64231334}"
ENVN="${INFISICAL_ENV:-dev}"
mkdir -p "$RUN"

die() { echo "demo_up: $*" >&2; exit 1; }
inf() { infisical run --projectId "$PROJECT" --env="$ENVN" --silent "$@"; }

# Live config (not secrets). Bank hashes are the frozen ta/en banks in stimuli/.
export SEVI_ENV="${SEVI_ENV:-dev}"
export SEVI_RESULT__BANK_DIR="${SEVI_RESULT__BANK_DIR:-stimuli}"
export SEVI_RESULT__ENABLED_LANGS="${SEVI_RESULT__ENABLED_LANGS:-ta,en}"
HASHES='{"ta":"3db5d7a4948d16a7b70d0cdfb43f68de465876047f269f90355a2c25234cd1e3","en":"d6b066bd96e413ca0bd73f48673bceaebc04e2e36353fc67a0a677dc1b6c7cc7"}'
export SEVI_RESULT__FROZEN_BANK_HASHES="${SEVI_RESULT__FROZEN_BANK_HASHES:-$HASHES}"
export SEVI_BROWSER_VOICE__ENABLED="${SEVI_BROWSER_VOICE__ENABLED:-true}"
export SEVI_PHONE__ENABLED="${SEVI_PHONE__ENABLED:-false}"
export SEVI_EVON__RESPONSE_FORMAT_STYLE="${SEVI_EVON__RESPONSE_FORMAT_STYLE:-openai}"
export SEVI_EVON__SCHEMA_PATTERN="${SEVI_EVON__SCHEMA_PATTERN:-false}"

ready() {
  inf --path /accounts -- sh -c \
    'SEVI_OP_USER=sevi-op SEVI_OP_PASS="$OPERATOR_PASSWORD" uv run sevi ready --min degraded' \
    2>/dev/null
}

serving() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

up() {
  command -v infisical >/dev/null || die "infisical CLI not installed"
  if [[ -f "$RUN/evon.env" ]]; then
    local evon
    evon="$(sed -n 's/^SEVI_EVON__BASE_URL=//p' "$RUN/evon.env" | head -1)"
    # Kaggle stops itself after 30 idle minutes; a stale env file must not count as "up".
    curl -sf --max-time 10 "${evon%/v1}/health" >/dev/null || rm -f "$RUN/evon.env"
  fi
  if [[ ! -f "$RUN/evon.env" ]]; then
    echo "starting Evon on Kaggle (about 6 min)..."
    scripts/evon_kaggle.sh push
    scripts/evon_kaggle.sh url
  fi
  scripts/ingress_up.sh funnel
  if serving; then
    echo "sevi serve already running (pid $(cat "$PIDF")); restarting to pick up config"
    down_serve
  fi
  # The page is served by sevi itself through the Funnel (Pages needs a paid plan on a private
  # repo), so the WebSocket origin check must accept the public URL.
  local origin
  origin="$(sed -n 's/^SEVI_PUBLIC_BASE_URL=//p' "$RUN/ingress.env" | head -1)"
  [[ -n "$origin" ]] || die "no public URL in $RUN/ingress.env"
  # Infisical's value wins over the caller's env, so the demo cap is applied inside the run.
  inf -- env SEVI_LEDGER__BUDGET_CREDITS_PER_KEY="${DEMO_BUDGET:-2500}" \
    SEVI_LEDGER__KEY_B_PUBLIC_STOP_SESSIONS="${DEMO_STOP_SESSIONS:-3}" \
    SEVI_PAGES_ORIGIN="$origin" make serve >"$LOG" 2>&1 &
  echo $! >"$PIDF"
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${SEVI_BIND_PORT:-8000}/healthz" >/dev/null && break
    sleep 2
  done
  serving || die "sevi serve exited; see $LOG"
  sleep 6
  status
}

status() {
  local out
  out="$(ready || true)"
  [[ -n "$out" ]] || die "readiness unavailable; see $LOG"
  python3 -c '
import json, sys
d = json.loads(sys.argv[1])
cr = d.get("credits", {})
print("admission:", d.get("admission_level"), "| evon:", d.get("evon"))
for k, v in cr.items():
    if isinstance(v, dict) and "public_key_low" in v:
        print("key", k, "remaining", v.get("budget_remaining"), "low", v.get("public_key_low"), "exposure_paused", v.get("exposure_paused"))
lvl = d.get("admission_level")
print("READY" if lvl == "L0" else "NOT READY (voice is offered only at L0)")
' "$out"
  if [[ -f "$RUN/ingress.env" ]]; then grep -h PUBLIC_BASE_URL "$RUN/ingress.env" | sed 's/^.*=/public: /;s#$#/sevi/#'; fi
}

down_serve() {
  if serving; then
    pkill -P "$(cat "$PIDF")" 2>/dev/null || true
    kill "$(cat "$PIDF")" 2>/dev/null || true
  fi
  pkill -f "sevi serve" 2>/dev/null || true
  rm -f "$PIDF"
}

down() {
  down_serve
  command -v tailscale >/dev/null && tailscale funnel reset >/dev/null 2>&1 || true
  [[ -f "$RUN/evon.env" ]] && scripts/evon_kaggle.sh stop || true
  echo "demo stopped"
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) die "usage: scripts/demo_up.sh up|down|status" ;;
esac
