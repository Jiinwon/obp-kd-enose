#!/usr/bin/env bash
set -e
python3 -m venv .venv || true
source .venv/bin/activate
pip install -U pip
pip install -r requirements_user.txt
python -m src.infer.user_demo --config configs/user_infer.yaml "$@"
