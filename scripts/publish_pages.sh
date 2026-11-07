#!/usr/bin/env bash
# Publish the static web client (web/static) as a site.
#
#   scripts/publish_pages.sh build [OUT_DIR]   copy web/static into a deployable folder
#                                              (default: dist/pages) for any static host
#   scripts/publish_pages.sh publish [REMOTE]  build, then commit the folder to the `gh-pages`
#                                              branch and push it (default remote: origin)
#
# GitHub Pages on a PRIVATE repo needs a paid plan. Without it, either deploy dist/pages to any
# static host, or skip Pages entirely: the app already serves web/static on the public listener
# at <SEVI_PUBLIC_BASE_URL>/sevi/ when SEVI_ENV != prod (see ops/ingress.md).
#
# Idempotent: an unchanged site makes no new gh-pages commit. Prints no secrets.
set -euo pipefail

ACTION="${1:-build}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/web/static"
BRANCH="gh-pages"

die() { echo "publish_pages: $*" >&2; exit 1; }

build() {
  local out="$1"
  [[ -f "$SRC/index.html" ]] || die "missing $SRC/index.html"
  rm -rf "$out"
  mkdir -p "$out"
  cp -R "$SRC/." "$out/"
  touch "$out/.nojekyll"
  echo "built: $out"
}

publish() {
  local remote="$1" tmp
  git -C "$ROOT" remote get-url "$remote" >/dev/null 2>&1 || die "no git remote '$remote'"
  tmp="$(mktemp -d)"
  trap 'git -C "$ROOT" worktree remove --force "$tmp" >/dev/null 2>&1 || rm -rf "$tmp"' EXIT
  if git -C "$ROOT" fetch -q "$remote" "$BRANCH" 2>/dev/null; then
    git -C "$ROOT" worktree add -q --detach "$tmp" FETCH_HEAD
  else
    git -C "$ROOT" worktree add -q --detach "$tmp"
    git -C "$tmp" checkout -q --orphan "$BRANCH"
  fi
  git -C "$tmp" rm -rq --ignore-unmatch . >/dev/null
  find "$tmp" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  build "$tmp" >/dev/null
  git -C "$tmp" add -A
  if git -C "$tmp" diff --cached --quiet && git -C "$tmp" rev-parse -q --verify HEAD >/dev/null; then
    echo "gh-pages: up to date"
    return
  fi
  git -C "$tmp" commit -qm "chore(pages): publish web/static ($(git -C "$ROOT" rev-parse --short HEAD))"
  git -C "$tmp" push -q "$remote" "HEAD:refs/heads/$BRANCH"
  echo "gh-pages: pushed to $remote/$BRANCH"
}

case "$ACTION" in
  build) build "${2:-$ROOT/dist/pages}" ;;
  publish) publish "${2:-origin}" ;;
  *) die "usage: $0 build [OUT_DIR] | publish [REMOTE]" ;;
esac
