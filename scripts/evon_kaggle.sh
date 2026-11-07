#!/usr/bin/env bash
# Evon (llama-server) on a free Kaggle 2x T4 session, for demo/video sessions only (issue #73).
#
#   scripts/evon_kaggle.sh push     render ops/evon/kaggle and push+run it (GPU T4 x2, internet)
#   scripts/evon_kaggle.sh status   kernel status, plus the tunnel URL once the server is up
#   scripts/evon_kaggle.sh url      wait for the URL (EVON_WAIT_S, default 1800 s), check
#                                   /health, write var/run/evon.env (loaded by `make serve`),
#                                   then one tiny warm-up completion (skip: EVON_SKIP_WARMUP=1)
#   scripts/evon_kaggle.sh stop     cancel the running session; remove var/run/evon.env
#
# URL hand-off: the kernel prints `SEVI_EVON_URL=https://<x>.trycloudflare.com` and publishes it to
# an ntfy.sh topic named by a hash of the key; this script polls that topic, falling back to the
# live session log (`kaggle kernels logs -f`; it returned HTTP 500 for running sessions
# in Nov 2026). Nothing on Kaggle needs a token or a reachable server of ours. Key: SEVI_EVON__API_KEY if set at push time (e.g. under
# `infisical run`), else a fresh random key per session kept in var/run/evon.key (0600) and
# written into evon.env. The key is never printed. Safety: the kernel stops itself after
# EVON_MAX_HOURS (default 2; free GPU quota is 6 h/week) or EVON_IDLE_MIN (default 30) minutes
# without a completion. At most 2 batch GPU sessions may exist at once: stop old ones first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/var/run"
PY=(python3 "$ROOT/scripts/evon_kaggle.py")
KEYF="$RUN/evon.key"
ENVF="$RUN/evon.env"
MAX_HOURS="${EVON_MAX_HOURS:-2}"
IDLE_MIN="${EVON_IDLE_MIN:-30}"
mkdir -p "$RUN"
umask 077

kuser() {
  [[ -n "${KAGGLE_USERNAME:-}" ]] && { echo "$KAGGLE_USERNAME"; return; }
  kaggle config view 2>/dev/null | sed -n 's/^- username: //p' | head -1
}
KID="$(kuser)/sevi-evon"

skey() {
  if [[ -f "$KEYF" ]]; then cat "$KEYF"; else echo "${SEVI_EVON__API_KEY:-}"; fi
}

# What the kernel announced so far: its ntfy.sh topic (key-derived; fast and independent of
# Kaggle's log stream), else the live session log (`logs -f`, cut after $1 seconds).
logs() {
  local t="${1:-20}" out="$RUN/evon-kernel.log" k
  k="$(skey)"
  if [[ -n "$k" ]]; then
    curl -fsS --max-time 15 "https://ntfy.sh/$(EVON_STOP_KEY="$k" "${PY[@]}" ntfy-topic)/json?poll=1&since=all" \
      2>/dev/null | "${PY[@]}" ntfy-log >"$out" || true
    grep -q SEVI_EVON_ "$out" && { cat "$out"; return; }
  fi
  kaggle kernels logs -f "$KID" >"$out" 2>&1 &
  local pid=$!
  for _ in $(seq "$t"); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  cat "$out"
}

state() { kaggle kernels status "$KID" 2>&1 | sed -n 's/.*status "\{0,1\}\([A-Za-z_.]*\)"\{0,1\}.*/\1/p' | head -1; }

case "${1:-}" in
  push)
    if [[ -n "${SEVI_EVON__API_KEY:-}" ]]; then
      export EVON_PUSH_KEY="$SEVI_EVON__API_KEY"; rm -f "$KEYF"
    else
      python3 -c 'import secrets; print(secrets.token_hex(24))' >"$KEYF"
      EVON_PUSH_KEY="$(cat "$KEYF")"; export EVON_PUSH_KEY
    fi
    rm -f "$ENVF"
    DEST="$RUN/evon-kernel"; rm -rf "$DEST"
    "${PY[@]}" render --src "$ROOT/ops/evon/kaggle" --dest "$DEST" --user "$(kuser)" \
      --tag-file "$ROOT/ops/evon/LLAMA_CPP_TAG" --max-hours "$MAX_HOURS" --idle-min "$IDLE_MIN" >/dev/null
    TIMEOUT_S="$(python3 -c "print(int(float('$MAX_HOURS') * 3600) + 900)")"
    out="$(kaggle kernels push -p "$DEST" --accelerator NvidiaTeslaT4 -t "$TIMEOUT_S" 2>&1 || true)"
    rm -rf "$DEST"  # the rendered copy holds the key
    echo "$out"
    if ! grep -q "successfully pushed" <<<"$out"; then
      echo "push failed (\"Maximum batch GPU session count\" = older sessions still running;" \
        "they stop themselves when idle, or cancel them on kaggle.com)" >&2
      exit 1
    fi
    echo "pushed $KID (hard cap ${MAX_HOURS} h, idle stop ${IDLE_MIN} min). Next: $0 url"
    ;;
  status)
    # Kaggle's status call returns an empty body (shown as QUEUED) while a batch session runs.
    st="$(state)"; echo "kernel $KID: ${st:-unknown}"
    if url="$(logs 20 | "${PY[@]}" parse-url)"; then echo "url: $url"; else echo "url: none"; fi
    ;;
  url)
    wait_s="${EVON_WAIT_S:-1800}"; t0=$SECONDS; url=""
    while (( SECONDS - t0 < wait_s )); do
      st="$(state)"
      case "$st" in
        *error*|*ERROR*|*cancel*|*CANCEL*|*complete*|*COMPLETE*)
          echo "kernel is $st (not running); last log lines:" >&2; logs 10 | tail -20 >&2; exit 1 ;;
      esac
      if url="$(logs 25 | "${PY[@]}" parse-url)"; then break; fi
      echo "waiting for the tunnel URL (${st:-queued}, $((SECONDS - t0)) s)..." >&2
      sleep 20
    done
    [[ -n "$url" ]] || { echo "no URL after ${wait_s} s" >&2; exit 1; }
    if [[ -f "$KEYF" ]]; then EVON_ENV_KEY="$(cat "$KEYF")"; export EVON_ENV_KEY; fi
    "${PY[@]}" write-env --url "$url" --out "$ENVF"
    if curl -fsS --max-time 15 "$url/health" >/dev/null; then
      echo "health: ok"
      # Warm the model so the first real session never hits the cold start (2/14 ReadTimeouts
      # on the first requests after boot). The key comes from evon.env and is never printed.
      if [[ "${EVON_SKIP_WARMUP:-0}" != 1 ]]; then
        "${PY[@]}" warmup --url "$url" --env-file "$ENVF" --timeout 120 || echo "warm-up: not ok" >&2
      fi
    else
      echo "health: not ok yet" >&2
    fi
    echo "url: $url"
    echo "wrote var/run/evon.env (SEVI_EVON__BASE_URL=$url/v1); restart \`make serve\` to use it"
    ;;
  stop)
    # The Kaggle API has no public cancel call: ask the kernel to stop itself through the tunnel
    # (a key-derived path its watchdog looks for), then wait for the session to end.
    url="$(logs 20 | "${PY[@]}" parse-url || true)"
    if [[ -n "$url" ]]; then
      K="$(skey)"; [[ -n "$K" ]] || { echo "no key: set SEVI_EVON__API_KEY" >&2; exit 1; }
      P="$(EVON_STOP_KEY="$K" "${PY[@]}" stop-path)"
      curl -s -o /dev/null --max-time 15 -d "SEVI_EVON_STOP=1" \
        "https://ntfy.sh/$(EVON_STOP_KEY="$K" "${PY[@]}" ntfy-topic)" || true
      curl -s -o /dev/null --max-time 15 -H "Authorization: Bearer $K" "$url$P" || true
      for _ in $(seq 30); do
        st="$(state)"; [[ "$st" =~ COMPLETE|ERROR|CANCEL|complete|error|cancel ]] && break; sleep 10
      done
    fi
    rm -f "$ENVF"
    st="$(state)"
    if [[ "$st" =~ COMPLETE|ERROR|CANCEL|complete|error|cancel ]]; then
      rm -f "$KEYF"; echo "stopped $KID ($st); removed var/run/evon.env"
    else
      echo "kernel still ${st:-unknown}: cancel it at https://www.kaggle.com/code/$KID" \
        "(it also stops itself after ${IDLE_MIN} idle min)" >&2; exit 1
    fi
    ;;
  *)
    echo "usage: $0 {push|status|url|stop}" >&2; exit 2 ;;
esac
