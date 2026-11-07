.PHONY: ingress pages pages-build shellcheck schema-snapshot web-test web-copy web-serve dev check check-slow lint fmt fmt-check type test cov serve live

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
	uv run pytest -m "not live and not slow"

# Slow tests (EV-ST-01, P-ST-04 ...): excluded from check, run separately.
check-slow:
	uv run pytest -m "slow and not live"

cov:
	uv run pytest -m "not live" --cov --cov-report=term-missing --cov-report=xml

# Public (:8000) + operator (:8001) listeners, one process. Secrets via Infisical in real use:
#   infisical run --env=dev -- make serve   (loads var/run/ingress.env from `make ingress` if present)
serve:
	set -a; [ -f var/run/ingress.env ] && . var/run/ingress.env; set +a; uv run sevi serve

# Expose the public listener: MODE=funnel (Tailscale, default) or MODE=quick (Cloudflare).
MODE ?= funnel
ingress:
	scripts/ingress_up.sh $(MODE)

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
