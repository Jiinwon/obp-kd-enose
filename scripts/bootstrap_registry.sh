#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: This script must be run inside a Git repository." >&2
  exit 1
fi

git checkout -B feature/data-registry

python -V; which python
python -m pip install --upgrade pip
pip install -e ".[dev]"

pre-commit install && pre-commit run --all-files
pytest -q || true

git add -A && git commit -m "feat(data): add common schema & data registry (step1)" || true
git push -u origin feature/data-registry || true

if command -v gh >/dev/null 2>&1; then
  gh pr create --base main --title "feat(data): common schema & data registry" --body "1단계: 레지스트리/데이터셋 스켈레톤 & 테스트" || true
else
  echo "gh CLI not found; skipping pull request creation." >&2
fi
