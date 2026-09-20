#!/usr/bin/env bash
# Standalone launcher: use the activated platform environment or an explicit Python.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
  cat <<'EOF'
Usage:
  bash run.sh a100|mi210|p150b|nvidia|amd|tenstorrent|cpu [run options]
  bash run.sh compare LEFT RIGHT --output DIR
  bash run.sh prepare [--inputs x-w-z.npz] --output FIXTURE_DIR

Activate an environment with NumPy and the selected accelerator runtime first.
Or set NORMBENCH_PYTHON=/absolute/path/to/python. No model or credentials needed.

Defaults: 2 fresh processes, 3 calls, FP32/BF16 stages, bundled frozen inputs.
Options: --output DIR --cases captured bf16-midpoints --precisions float32
         --repeats 5 --processes 2 --tt-device-id 0 --dry-run
         --control vllm (optional NVIDIA/AMD production-kernel comparison)
EOF
}
command="${1:---help}"
if [[ "$command" == --help || "$command" == -h ]]; then usage; exit 0; fi
shift
case "$command" in
  a100|nvidia|rtx3060) backend=nvidia ;;
  mi210|amd) backend=amd ;;
  p150b|tenstorrent) backend=tenstorrent ;;
  cpu) backend=cpu ;;
  compare|prepare) ;;
  *) usage >&2; exit 2 ;;
esac
python="${NORMBENCH_PYTHON:-python3}"
if ! command -v "$python" >/dev/null 2>&1; then
  echo "Python not found: $python. Activate your platform environment or set NORMBENCH_PYTHON." >&2
  exit 1
fi
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export NORMBENCH_CACHE_DIR="${NORMBENCH_CACHE_DIR:-$root/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$NORMBENCH_CACHE_DIR/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$NORMBENCH_CACHE_DIR/torchinductor}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "$command" == compare || "$command" == prepare ]]; then
  exec "$python" -m normbench "$command" "$@"
fi
exec "$python" -m normbench run --backend "$backend" "$@"
