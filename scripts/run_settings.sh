#!/usr/bin/env bash
# Usage: bash scripts/run_settings.sh --config suites/rtx3060.json --output results/my-settings
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
exec python -m driftbench_runner suite "$@"
