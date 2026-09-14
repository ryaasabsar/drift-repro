#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
DRIFTBENCH_RESULT_DIR="${1:-results/rtx3060-qwen35-vllm}"
DRIFTBENCH_CONFIG="${2:-configs/rtx3060-qwen35-vllm-batch16.json}"
mkdir -p "$DRIFTBENCH_RESULT_DIR"
# Serialize this project's GPU jobs across inference and safety evaluation.
exec 9>results/.gpu-experiment.lock
flock -n 9 || { echo 'Another run_full.sh job is using the GPU.' >&2; exit 1; }
DRIFTBENCH_RESUME=()
if [[ -f "$DRIFTBENCH_RESULT_DIR/manifest.json" ]]; then
  DRIFTBENCH_RESUME=(--resume)
fi
python -u -m driftbench_runner run --config "$DRIFTBENCH_CONFIG" \
  --output "$DRIFTBENCH_RESULT_DIR" "${DRIFTBENCH_RESUME[@]}"
python -u -m driftbench_runner judge-safety "$DRIFTBENCH_RESULT_DIR" \
  --device cuda --output "$DRIFTBENCH_RESULT_DIR/safety-labels.jsonl"
python -u -m driftbench_runner evaluate "$DRIFTBENCH_RESULT_DIR" --code \
  --safety-labels "$DRIFTBENCH_RESULT_DIR/safety-labels.jsonl"
python -m driftbench_runner status "$DRIFTBENCH_RESULT_DIR" --json
