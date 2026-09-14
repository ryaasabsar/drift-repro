#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
runner_python=.venv-tt-vllm/bin/python
if [[ ! -x "$runner_python" ]]; then
  for argument in "$@"; do
    if [[ "$argument" == --dry-run || "$argument" == --help || "$argument" == -h ]]; then
      exec .venv/bin/python -m driftbench_runner.blackhole_p150b "$@"
    fi
  done
  echo 'Missing .venv-tt-vllm/bin/python. Prepare a compatible TT-Metal/TTNN vLLM environment and install this runner with its client/evaluation dependencies there.' >&2
  exit 1
fi
exec "$runner_python" -m driftbench_runner.blackhole_p150b "$@"
