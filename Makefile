.PHONY: evon-up evon-stop evon-g4 ingress pages pages-build shellcheck schema-snapshot web-test web-copy web-serve dev check check-slow lint fmt fmt-check type test cov serve demo e2e live

TYPED := src/sevi/contracts src/sevi/core src/sevi/app src/sevi/adapters src/sevi/transport src/sevi/fakes src/sevi/cli.py

dev:
	uv sync
	uv run pre-commit install

check: lint fmt-check type test

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

fmt-check:
	uv run ruff format --check .

type:
	uv run mypy --strict $(TYPED) --exclude "src/sevi/fakes/scenarios/"

test:
	uv run pytest -m "not live and not slow and not e2e"

# Whole-app E2E with fakes (#20): the real app on uvicorn (127.0.0.1) + a scripted browser over
# a real WebSocket; no network beyond localhost, no keys. Excluded from `check` (about 90 s).
e2e:
	uv run pytest -m "e2e and not live" -v tests/integration/a/test_e2e_app.py

# Slow tests (EV-ST-01, P-ST-04 ...): excluded from check, run separately.
check-slow:
	uv run pytest -m "slow and not live"

cov:
	uv run pytest -m "not live" --cov --cov-report=term-missing --cov-report=xml

# Public (:8000) + operator (:8001) listeners, one process. Secrets via Infisical in real use:
#   infisical run --env=dev -- make serve   (loads var/run/ingress.env from `make ingress`
#   and var/run/evon.env from `make evon-up`, if present)
# `SEVI_DEMO_MODE=1 make serve` (or `make demo`): a full local test with no keys (fake Prisma and
# Evon, the synthetic test bank, placeholder prompts, demo-only credit figures); any variable
# already set wins. Open http://127.0.0.1:8000/sevi/ (127.0.0.1: the page origin must match).
DEMO_DEFAULTS = if [ "$$SEVI_DEMO_MODE" = 1 ] || [ "$$SEVI_DEMO_MODE" = true ]; then \
	  export SEVI_ENV="$${SEVI_ENV:-demo}" \
	    SEVI_RESULT__BANK_DIR="$${SEVI_RESULT__BANK_DIR:-stimuli/test-bank}" \
	    SEVI_RESULT__ENABLED_LANGS="$${SEVI_RESULT__ENABLED_LANGS:-ta}" \
	    SEVI_RESULT__ALLOW_UNFROZEN_BANK="$${SEVI_RESULT__ALLOW_UNFROZEN_BANK:-true}" \
	    SEVI_BROWSER_VOICE__ENABLED="$${SEVI_BROWSER_VOICE__ENABLED:-true}" \
	    SEVI_LEDGER__CREDITS_PER_STT_S="$${SEVI_LEDGER__CREDITS_PER_STT_S:-1}" \
	    SEVI_LEDGER__BUDGET_CREDITS_PER_KEY="$${SEVI_LEDGER__BUDGET_CREDITS_PER_KEY:-100000}"; \
	fi

serve:
	set -a; [ -f var/run/ingress.env ] && . var/run/ingress.env; [ -f var/run/evon.env ] && . var/run/evon.env; set +a; $(DEMO_DEFAULTS); uv run sevi serve

demo:
	SEVI_DEMO_MODE=1 $(MAKE) serve

# Expose the public listener: MODE=funnel (Tailscale, default) or MODE=quick (Cloudflare).
MODE ?= funnel
ingress:
	scripts/ingress_up.sh $(MODE)

# Evon on a free Kaggle 2x T4 session (demo/video only): push the kernel, wait for the tunnel
# URL, write var/run/evon.env (then restart `make serve`). Stop it after use: `make evon-stop`.
evon-up:
	scripts/evon_kaggle.sh push
	scripts/evon_kaggle.sh url

evon-stop:
	scripts/evon_kaggle.sh stop

evon-g4:
	uv run python scripts/evon_g4.py

# Publish web/static to the gh-pages branch; `make pages-build` only builds dist/pages.
pages:
	scripts/publish_pages.sh publish

pages-build:
	scripts/publish_pages.sh build

shellcheck:
	shellcheck scripts/*.sh

# Real API calls. Never run by CI or builder agents. Secrets come from Infisical only.
live:
	SEVI_ALLOW_LIVE=1 infisical run --env=dev -- uv run pytest -m live

# Regenerate the C-CON-01 contract schema snapshot after an intentional contract change.
schema-snapshot:
	SEVI_UPDATE_SCHEMA_SNAPSHOT=1 uv run pytest -q tests/contract/test_schema_snapshot.py

# Browser client (web/static): JS unit tests (node >= 20) and the generated copy tables.
web-test:
	node --test web/tests/*.test.mjs
	uv run pytest -q tests/unit/b/test_web.py

web-copy:
	uv run python scripts/build_web_copy.py

web-serve:
	uv run python -m http.server 8080 --directory web/static
