#!/usr/bin/env bash
# Workspace-local MI210 serving installation, following the NVIDIA/TT installer roles.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$root/.venv-rocm-vllm"
uv="$root/.tools/uv"
index=https://wheels.vllm.ai/rocm/0.17.1/rocm700
dry_run=false
check_only=false
check_gpu=false

usage() {
    cat <<'EOF'
Usage: bash scripts/install_rocm_vllm.sh [--dry-run | --check | --check-gpu]

Install .venv-rocm-vllm with managed Python 3.12.14, vLLM 0.17.1+rocm700
and its matching PyTorch 2.9.1 ROCm build. Requires Linux x86_64, glibc >=2.35
and a ROCm 7.0 host. Keeps the same tokenizer pins as NVIDIA.

--dry-run    Print commands without creating files or downloading packages.
--check      Check an existing installation without installing or opening GPUs.
--check-gpu  Check an existing installation and calculate on each visible GPU.
             Run inside your allocated MI210 job; keeps scheduler visibility.

Prepare the benchmark client separately: bash scripts/bootstrap.sh --client-only
No model downloads, CUDA toolkit installation, driver changes or system packages.
EOF
}
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --check) check_only=true ;;
        --check-gpu) check_only=true; check_gpu=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'The ROCm wheels require Linux x86_64.' >&2; exit 1
fi
glibc="$(getconf GNU_LIBC_VERSION)"
if [[ ! "$glibc" =~ ^glibc\ ([0-9]+)\.([0-9]+)$ ]] ||
   (( BASH_REMATCH[1] < 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] < 35) )); then
    echo "The ROCm wheels require glibc >=2.35; found $glibc." >&2; exit 1
fi
if [[ -L "$target" || ( -e "$target" && ! -f "$target/pyvenv.cfg" ) ]]; then
    echo "Refusing symlink/non-venv target: $target" >&2; exit 1
fi
if $check_only && ! $dry_run && [[ ! -x "$target/bin/python" || ! -x "$uv" ]]; then
    echo 'Installation missing; run bash scripts/install_rocm_vllm.sh first.' >&2; exit 1
fi

# Source the same cache settings as NVIDIA; use explicit interpreters throughout.
source "$root/scripts/env.sh"
unset VIRTUAL_ENV CONDA_PREFIX UV_INDEX UV_DEFAULT_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_INDEX_STRATEGY
export PYTHONPATH="$root"
run() {
    printf '+ '; printf '%q ' "$@"; printf '\n'
    if ! $dry_run; then "$@"; fi
}

if ! $check_only; then
    run bash "$root/scripts/ensure_uv.sh"
    if ! $dry_run; then
        # Serialize installers sharing native package caches.
        exec 9>"$root/.tools/framework-install.lock"
        flock 9
    fi
    if [[ ! -e "$target" ]]; then
        run "$uv" --no-config venv --managed-python --python 3.12.14 "$target"
    fi
    run "$target/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected Python 3.12; move the old .venv-rocm-vllm aside and rerun"'
    run "$uv" --no-config pip sync --python "$target/bin/python" \
        --default-index https://pypi.org/simple --extra-index-url "$index" \
        --index-strategy first-index --only-binary vllm,torch \
        "$root/requirements.rocm-vllm.lock.txt"
    run "$uv" --no-config pip install --python "$target/bin/python" --no-deps -e "$root"
fi

run "$uv" --no-config pip check --python "$target/bin/python"
if $dry_run; then
    echo '+ Check Python, pinned packages, vLLM import and ROCm PyTorch runtime (timeout 90s).'
    if $check_gpu; then echo '+ Check calculations and Triton initialization on visible MI210 GPUs (timeout 120s).'; fi
    exit 0
fi

timeout 90s "$target/bin/python" -u - "$root" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
from driftbench_runner.software import check_runtime

assert sys.version_info[:2] == (3, 12), 'Expected Python 3.12'
for line in (Path(sys.argv[1]) / 'requirements.rocm-vllm.lock.txt').read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, expected = line.split('==')
    actual = metadata.version(name)
    assert actual == expected, f'{name}: expected {expected}, found {actual}'
for name in ('vllm', 'torch', 'transformers', 'tokenizers'):
    print(f'{name}: {metadata.version(name)}')
import torch
import vllm
import vllm._C
assert torch.version.hip and torch.version.hip.startswith('7.0'), f'Expected ROCm 7.0 PyTorch, found HIP={torch.version.hip}'
assert torch.version.cuda is None, 'Unexpected CUDA PyTorch build'
report = check_runtime({'hardware': {'vendor': 'amd'}, 'backend': 'vllm'}, {
    'python': sys.version.split()[0], 'rocm_runtime': torch.version.hip,
    'packages': {name: metadata.version(name) for name in ('vllm', 'torch', 'transformers', 'tokenizers')},
})
assert report['status'] == 'match', report
print('Package/native import checks passed. ROCm runtime:', torch.version.hip)
PY

if ! $check_only; then
    "$uv" --no-config pip freeze --python "$target/bin/python" > "$target/installed-packages.txt"
fi
if $check_gpu; then
    timeout 120s "$target/bin/python" -u - <<'PY'
import torch
import triton
assert torch.cuda.is_available(), 'No usable ROCm GPU; check your job allocation and device permissions'
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, 'gcnArchName', '')
        print(f'GPU {index}: {props.name}, architecture={arch}')
        assert arch.split(':')[0] == 'gfx90a', 'This check targets MI210/gfx90a'
        x = torch.ones(1, device=f'cuda:{index}')
        assert (x + 1).item() == 2.0
        triton.runtime.driver.active.get_current_device()
        print('GPU calculation and Triton initialization passed')
PY
else
    echo 'GPU execution not tested. Run: bash scripts/install_rocm_vllm.sh --check-gpu'
fi
echo 'Model smoke run: bash scripts/run_mi210.sh --framework vllm --limit 2 --run-id mi210-vllm-smoke'
