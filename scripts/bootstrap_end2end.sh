#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python3}

log() {
  echo "[bootstrap] $*"
}

log "Ensuring dependencies are installed"
$PYTHON_BIN -m pip install --upgrade pip >/dev/null
$PYTHON_BIN -m pip install -e . >/dev/null
if ! $PYTHON_BIN - <<'PY' >/dev/null 2>&1
import torch
PY
then
  log "Installing CPU-only torch"
  $PYTHON_BIN -m pip install torch --extra-index-url https://download.pytorch.org/whl/cpu >/dev/null
else
  log "torch already available"
fi

log "Generating docking prior table"
$PYTHON_BIN docking/scripts/make_prior_table.py --cfg configs/prior/prior_config.yaml

log "Running forward-chaining experiment"
$PYTHON_BIN -m src.runner.run_experiment --cfg configs/experiment/uci_forward.yaml

log "Building experiment report"
$PYTHON_BIN tools/make_report.py --metrics results/uci_forward_mvp/metrics.csv --output results/uci_forward_mvp/report.md --experiment uci_forward_mvp

log "Done"
