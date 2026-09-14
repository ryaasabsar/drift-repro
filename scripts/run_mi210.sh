#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
runner_python=.venv-rocm-vllm/bin/python
if [[ ! -x "$runner_python" ]]; then
  # Only plan/help mode may use the local NVIDIA/client environment.
  for argument in "$@"; do
    if [[ "$argument" == --dry-run || "$argument" == --help || "$argument" == -h ]]; then
      exec .venv/bin/python -m driftbench_runner.mi210 "$@"
    fi
  done
  echo 'Missing .venv-rocm-vllm/bin/python. Prepare ROCm PyTorch/vLLM and install this runner with its client/evaluation dependencies there.' >&2
  exit 1
fi
exec "$runner_python" -m driftbench_runner.mi210 "$@"
