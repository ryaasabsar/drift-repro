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
    uv venv --managed-python --python 3.12 .venv-client
    uv pip install --python .venv-client/bin/python \
      'transformers==4.57.6' 'huggingface-hub==0.36.2' 'sentencepiece==0.2.2'
    uv pip install --python .venv-client/bin/python --no-deps -e .
  else
    uv venv --managed-python --python 3.12 .venv
    uv pip sync --python .venv/bin/python requirements.lock.txt
    uv pip install --python .venv/bin/python --no-deps -e .
  fi
fi
if [[ ! -d vendor/driftbench-ae ]]; then
  git clone https://github.com/GianluigiVitale/driftbench-ae.git vendor/driftbench-ae
fi
git -C vendor/driftbench-ae checkout c915e781a17c2d4c9bd768e60a1bc9734c5f2895
echo 'Ready. Bubblewrap is needed only on the RTX code-evaluation host.'
