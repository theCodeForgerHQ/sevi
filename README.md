# Sevi

காது சோதனை

Sevi is a phone-quality digits-in-noise hearing screen built on Gnani speech models: it plays spoken digit triplets in noise, adapts the noise level to the listener's answers, and reports a speech reception threshold with plain-language guidance.

## Development

```sh
make dev      # install deps and pre-commit hooks
make check    # lint, format check, mypy --strict, tests (no live calls)
```

Secrets are never stored in the repo. Anything that needs them runs under Infisical:

```sh
infisical run --env=dev -- uv run <command>
```
