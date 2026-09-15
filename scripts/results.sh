#!/usr/bin/env bash
set -euo pipefail
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$script_root/scripts/env.sh"
case "${1:-}" in
  stage|compare|compare-many|serve) default_python="$script_root/.venv/bin/python" ;;
  *) default_python="$script_root/.venv-client/bin/python" ;;
esac
if [[ "${1:-}" == stage && "${2:-}" == status ]]; then default_python="$script_root/.venv-client/bin/python"; fi
runner_python="${DRIFTBENCH_PYTHON:-$default_python}"
if [[ ! -x "$runner_python" ]]; then
  echo "Missing runner Python: $runner_python. Install the common client with bash scripts/bootstrap.sh --client-only; evaluation uses the NVIDIA environment from bash scripts/bootstrap.sh nvidia." >&2
  exit 1
fi
export PYTHONPATH="$script_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$runner_python" -m driftbench_runner "$@"
