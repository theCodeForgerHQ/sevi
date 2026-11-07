# Ingress runbook

Sevi runs as one process with two listeners (`make serve` / `sevi serve`):

| listener | bind | exposed? |
|---|---|---|
| public (`/api/v1/*`, `/ws/*`, `/readyz`, `/sevi/` static) | `SEVI_BIND_HOST:SEVI_BIND_PORT` (127.0.0.1:8000) | yes, via a tunnel |
| operator (`/op/api/*`) | `SEVI_OP_BIND_HOST:SEVI_OP_BIND_PORT` (127.0.0.1:8001) | **never**; Tailscale/loopback only |

## 1. Start the app

```sh
infisical run --env=prod -- make serve      # secrets only ever come from Infisical
```

## 2. Open the tunnel

```sh
make ingress                 # Tailscale Funnel (default): tailscale funnel --bg 8000
make ingress MODE=quick      # fallback: Cloudflare quick tunnel (cloudflared tunnel --url)
```

The script is safe to re-run: it reuses a live funnel/tunnel. It writes the public URL into
`web/static/config.json` and prints three values. Put them in Infisical (or the env) and restart:

```
SEVI_PUBLIC_BASE_URL=https://<host>.ts.net        # or https://<random>.trycloudflare.com
SEVI_CLIENT_IP_HEADER=X-Forwarded-For             # quick tunnel: Cf-Connecting-Ip
SEVI_TRUSTED_PROXIES=127.0.0.1/32,::1/128         # both tunnels connect from loopback
```

The quick-tunnel URL changes every time `cloudflared` restarts. Re-run `make ingress MODE=quick`,
then republish the site (step 3); open pages re-read `config.json` on a failed request.
Stop it with `kill "$(cat var/ingress/cloudflared.pid)"`. Turn the funnel off with
`tailscale funnel reset`.

## 3. Publish the web client

- **GitHub Pages:** `make pages` pushes `web/static` to the `gh-pages` branch (no commit if
  unchanged). Pages on a **private** repo needs a paid plan. Set `SEVI_PAGES_ORIGIN` to the
  Pages origin so CORS allows it.
- **Any static host:** `make pages-build` writes `dist/pages/`. Upload that folder.
- **No separate host (simplest when Pages is unavailable):** the public listener already serves
  `web/static` at `<SEVI_PUBLIC_BASE_URL>/sevi/` (same origin, so no CORS). Share that link.
  It is mounted only when `SEVI_ENV != prod`, so use `SEVI_ENV=demo` for this mode.

## 4. Check

```sh
export SEVI_OP_USER=sevi-op SEVI_OP_PASS=...   # type it; never put it in shell history files
sevi gate                     # exit 0 iff /op/api/readyz is ok (--min degraded to relax)
curl -fsS "$SEVI_PUBLIC_BASE_URL/readyz"
sevi ledger show              # credits per key; `sevi ledger local-summary` works with the app down
```

## Operator CLI quick reference

`sevi switch <evon|prisma|timbre_runtime|phone|browser_voice|demo_mode|friends> <on|off>`,
`sevi station [--lang ta] [--mode keypad]`, `sevi dial P-01` (needs the phone switch),
`sevi allowlist list|add P-01`, `sevi ledger reconcile --key A --bucket DEV --dashboard-used N`,
`sevi tail <session_id> [-f]`. `SEVI_OP_URL` overrides the operator URL (default
`http://127.0.0.1:8001`).

## Incidents

- **Credits running low:** `sevi switch browser_voice off` and `sevi switch prisma off`; the
  ledger also refuses on its own at `SEVI_LEDGER__BUDGET_CREDITS_PER_KEY`.
- **Abuse / load:** `make ingress` is the only public path; `tailscale funnel reset`
  (or kill `cloudflared`) takes the site offline without stopping in-flight operator work.
