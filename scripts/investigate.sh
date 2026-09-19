#!/usr/bin/env bash
set -euo pipefail
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$script_root/scripts/env.sh"
runner_python="${DRIFTBENCH_PYTHON:-$script_root/.venv-client/bin/python}"
if [[ ! -x "$runner_python" ]]; then
  echo "Missing client Python: $runner_python. Run bash scripts/bootstrap.sh --client-only or set DRIFTBENCH_PYTHON." >&2
  exit 1
fi
export PYTHONPATH="$script_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$runner_python" -m driftbench_runner.investigation "$@"
