.PHONY: schema-snapshot dev check check-slow lint fmt fmt-check type test cov serve live

TYPED := src/sevi/contracts src/sevi/core

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
	uv run mypy --strict $(TYPED)

test:
	uv run pytest -m "not live and not slow"

# Slow tests (EV-ST-01, P-ST-04 ...): excluded from check, run separately.
check-slow:
	uv run pytest -m "slow and not live"

cov:
	uv run pytest -m "not live" --cov --cov-report=term-missing --cov-report=xml

serve:
	@echo "serve: not implemented yet (app lands in a later PR)"

# Real API calls. Never run by CI or builder agents. Secrets come from Infisical only.
live:
	SEVI_ALLOW_LIVE=1 infisical run --env=dev -- uv run pytest -m live

# Regenerate the C-CON-01 contract schema snapshot after an intentional contract change.
schema-snapshot:
	SEVI_UPDATE_SCHEMA_SNAPSHOT=1 uv run pytest -q tests/contract/test_schema_snapshot.py
