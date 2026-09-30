#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/autodl_env.sh"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
python3 -m venv "$PROJECT_ROOT/.venv"

if [[ "${1:-}" == "--wheelhouse" ]]; then
  WHEELHOUSE="${2:?usage: bootstrap_autodl.sh --wheelhouse /root/autodl-tmp/wheels}"
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" "setuptools>=68" wheel
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$PROJECT_ROOT/requirements-reboot.txt"
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"
else
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/requirements-reboot.txt"
  "$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT"
fi

"$PROJECT_ROOT/.venv/bin/python" -m unittest discover -s "$PROJECT_ROOT/tests" -v
printf 'Bootstrap passed. In every new shell run: source %q\n' "$PROJECT_ROOT/scripts/autodl_env.sh"
