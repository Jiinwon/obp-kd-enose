#!/usr/bin/env bash
# One-stop script for end users to run inference locally.
set -e

VENV=.venv
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --quiet -r requirements_user.txt

# Placeholder for downloading release bundle. In practice replace URL.
BUNDLE_URL=${BUNDLE_URL:-https://example.com/release.tar.gz}
RELEASE_DIR=release
mkdir -p "$RELEASE_DIR"
# curl -L "$BUNDLE_URL" | tar -xz -C "$RELEASE_DIR"  # commented placeholder

python -m src.infer.user_demo --config configs/user_infer.yaml