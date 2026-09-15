#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DRIFTBENCH_PYTHON="${DRIFTBENCH_PYTHON:-$root/.venv-client/bin/python}"
exec bash "$root/scripts/results.sh" infer --config "$root/suites/mi210.json" "$@"
