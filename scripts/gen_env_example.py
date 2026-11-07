"""Regenerate `.env.example` from `sevi.app.settings.Settings` (U-CFG-01 checks they agree)."""

from __future__ import annotations

from pathlib import Path

from sevi.app.settings import render_env_example


def main() -> None:
    path = Path(__file__).resolve().parents[1] / ".env.example"
    path.write_text(render_env_example(), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
