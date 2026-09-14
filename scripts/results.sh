#!/usr/bin/env bash
set -euo pipefail
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$script_root/scripts/env.sh"
runner_python="${DRIFTBENCH_PYTHON:-$script_root/.venv/bin/python}"
if [[ ! -x "$runner_python" ]]; then
  echo 'Missing runner Python. Set DRIFTBENCH_PYTHON to an environment with the runner dependencies.' >&2
  exit 1
fi
export PYTHONPATH="$script_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$runner_python" -m driftbench_runner "$@"
