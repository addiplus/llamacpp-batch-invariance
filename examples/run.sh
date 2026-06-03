#!/usr/bin/env bash
# examples/run.sh -- run the batch-invariance gate against the bundled mock (no GPU).
#
# Two modes:
#   ./run.sh             run the gate against the in-process mock: a divergent run goes RED,
#                        an honest run reaches a (non-promotable) green status. Also runs the
#                        mock's own zero-arg --demo first.
#   ./run.sh --dry-run   print the plan + KV/footprint projection and launch NOTHING.
#
# The canonical user-facing entry point is the installed console script:
#   batch-invariance run-mock --out-dir ./certs                 # writes a GREEN cert
#   batch-invariance run-mock --out-dir ./certs --batch-divergence   # same gate, now RED
# This script runs straight from a checkout (no install needed) via examples/run_gate.py,
# which is a thin wrapper over the SAME batch_invariance.cli.run_mock_gate core.
#
# Needs only Python 3.10+ and this repo (stdlib only -- no GPU, no model, no real server).
# Resolves Python as: $PYTHON env override, else python3, else python.
set -euo pipefail

# --- locate the repo root (this script lives in examples/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- pick a Python interpreter ---
if [ -n "${PYTHON:-}" ]; then
  PY="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "error: no python3/python on PATH (set \$PYTHON to your interpreter)" >&2
  exit 1
fi

# Make `batch_invariance` (src/) and `examples.scorer` importable when run from a checkout.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

# --- --dry-run: preview only, launch nothing ---
for arg in "$@"; do
  if [ "${arg}" = "--dry-run" ]; then
    echo ">> batch-invariance example (DRY RUN -- launches nothing)"
    exec "${PY}" "${SCRIPT_DIR}/run_gate.py" --dry-run
  fi
done

# --- 1) the mock's own zero-arg RED-proof demonstration (honest vs --batch-divergence) ---
echo ">> [demo] mock_llama_server --demo (serial vs concurrent, honest vs divergent)"
"${PY}" -m batch_invariance.mock_llama_server --demo
echo

# --- 2) drive the REAL A/B/C gate against the mock (RED on divergence, non-promotable green) ---
echo ">> [gate] running the A/B/C batch-invariance gate against the mock"
exec "${PY}" "${SCRIPT_DIR}/run_gate.py" "$@"
