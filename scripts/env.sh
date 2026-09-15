#!/usr/bin/env bash
# Source from any directory; keep model and compiler caches in this workspace.
DRIFTBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$DRIFTBENCH_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$DRIFTBENCH_ROOT/.cache"
export UV_CACHE_DIR="$DRIFTBENCH_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$DRIFTBENCH_ROOT/.tools/python"
export VLLM_CACHE_ROOT="$DRIFTBENCH_ROOT/.cache/vllm"
export TRITON_CACHE_DIR="$DRIFTBENCH_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$DRIFTBENCH_ROOT/.cache/torchinductor"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PATH="$DRIFTBENCH_ROOT/.venv/bin:$DRIFTBENCH_ROOT/.tools:$PATH"
