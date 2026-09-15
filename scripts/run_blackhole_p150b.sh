#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$root/.venv-tt-vllm/activate-tt.sh" ]]; then source "$root/.venv-tt-vllm/activate-tt.sh"; fi
export DRIFTBENCH_PYTHON="${DRIFTBENCH_PYTHON:-$root/.venv-tt-vllm/bin/python}"
exec bash "$root/scripts/results.sh" infer --config "$root/suites/blackhole-p150b.json" "$@"
