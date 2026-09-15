#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner_override="${DRIFTBENCH_PYTHON:-$root/.venv-client/bin/python}"
# Older activation files selected the serving interpreter as the benchmark client.
if [[ "$runner_override" == "$root/.venv-tt-vllm/bin/python" ]]; then runner_override="$root/.venv-client/bin/python"; fi
if [[ -f "$root/.venv-tt-vllm/activate-tt.sh" ]]; then source "$root/.venv-tt-vllm/activate-tt.sh"; fi
export DRIFTBENCH_PYTHON="$runner_override"
exec bash "$root/scripts/results.sh" infer --config "$root/suites/blackhole-p150b.json" "$@"
