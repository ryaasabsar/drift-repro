#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv. Run bash scripts/bootstrap.sh first.' >&2
  exit 1
fi
exec .venv/bin/python -m driftbench_runner.a100 "$@"
