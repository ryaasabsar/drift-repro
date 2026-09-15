#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-nvidia}"
case "$mode" in
  nvidia|--client-only|--datasets-only) ;;
  *) echo 'Usage: bash scripts/bootstrap.sh [nvidia|--client-only|--datasets-only]' >&2; exit 2 ;;
esac
mkdir -p .tools vendor
source scripts/env.sh
if [[ "$mode" != --datasets-only ]]; then
  source scripts/ensure_uv.sh
  if [[ "$mode" == --client-only ]]; then
    # No torch/CUDA/ROCm wheels: a tokenizer client can drive any serving host.
    if [[ -L .venv-client || ( -e .venv-client && ! -f .venv-client/pyvenv.cfg ) ]]; then
      echo 'Refusing to replace a symlink or non-venv .venv-client directory.' >&2; exit 1
    fi
    if [[ ! -x .venv-client/bin/python ]] || [[ "$(.venv-client/bin/python -c 'import platform; print(platform.python_version())')" != 3.12.14 ]]; then
      echo 'Preparing the dedicated client environment with Python 3.12.14.'
      uv venv --managed-python --python 3.12.14 --clear .venv-client
    fi
    uv pip sync --python .venv-client/bin/python requirements.client.lock.txt
    uv pip install --python .venv-client/bin/python --no-deps -e .
  else
    if [[ ! -x .venv/bin/python ]]; then uv venv --managed-python --python 3.12.14 .venv; fi
    uv pip sync --python .venv/bin/python requirements.lock.txt
    uv pip install --python .venv/bin/python --no-deps -e .
  fi
fi
if [[ ! -d vendor/driftbench-ae ]]; then
  git clone https://github.com/GianluigiVitale/driftbench-ae.git vendor/driftbench-ae
fi
git -C vendor/driftbench-ae checkout c915e781a17c2d4c9bd768e60a1bc9734c5f2895
echo 'Ready. Bubblewrap is needed only on the RTX code-evaluation host.'
