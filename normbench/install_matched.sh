#!/usr/bin/env bash
# Dedicated, pinned environments for the A100/MI210 numerical experiment.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
backend="${1:---help}"
mode="${2:-install}"
usage() {
  echo 'Usage: bash normbench/install_matched.sh nvidia|amd [--dry-run|--check]'
  echo 'Installs Python 3.12.14, PyTorch 2.9.1, Triton 3.5.1, NumPy 2.2.6.'
  echo 'Run inside an allocated GPU job; --check rechecks without installing.'
}
if [[ "$backend" == --help || "$backend" == -h ]]; then usage; exit 0; fi
if [[ $# -gt 2 || ! "$backend" =~ ^(nvidia|amd)$ || ! "$mode" =~ ^(install|--dry-run|--check)$ ]]; then
  usage >&2; exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'These wheels require Linux x86_64.' >&2; exit 1
fi
target="$root/.venv-matched-$backend"
lock="$root/requirements.matched-$backend.lock.txt"
if [[ "$mode" == --dry-run ]]; then
  echo "Environment: $target"
  echo "Lockfile: $lock"
  echo 'Managed Python: 3.12.14; package hashes and GPU/Triton execution checked.'
  echo 'Runtime: NVIDIA CUDA 12.8 / AMD ROCm 6.4; existing node drivers used.'
  exit 0
fi
if [[ -L "$target" || ( -e "$target" && ! -f "$target/pyvenv.cfg" ) ]]; then
  echo "Refusing non-venv or symlink target: $target" >&2; exit 1
fi
export UV_CACHE_DIR="$root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/.tools/python"
export TRITON_CACHE_DIR="$root/.cache/matched-$backend/triton"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 PYTHONHASHSEED=42
uv_tool="$root/../.tools/uv"
if [[ ! -x "$uv_tool" ]]; then uv_tool="$root/.tools/uv"; fi
if [[ ! -x "$uv_tool" ]]; then
  if [[ "$mode" == --check ]]; then echo 'Run the installer first.' >&2; exit 1; fi
  mkdir -p "$root/.tools"
  curl -fsSL https://astral.sh/uv/0.12.11/install.sh -o "$root/.tools/install-uv.sh"
  UV_UNMANAGED_INSTALL="$root/.tools" sh "$root/.tools/install-uv.sh"
fi
python="$target/bin/python"
if [[ "$mode" == install ]]; then
  if [[ ! -e "$target" ]]; then
    "$uv_tool" venv --managed-python --python 3.12.14 "$target"
  fi
  "$python" -c 'import platform; assert platform.python_version() == "3.12.14", "Expected Python 3.12.14"'
  "$uv_tool" pip sync --python "$python" --require-hashes --only-binary :all: "$lock"
fi
"$uv_tool" pip check --python "$python"
# Reuse the workspace compiler when present; managed Python supplies headers.
if [[ -x "$root/../.tools/gcc13/bin/x86_64-conda-linux-gnu-gcc" ]]; then
  export CC="$root/../.tools/gcc13/bin/x86_64-conda-linux-gnu-gcc"
fi
timeout 120s "$python" - "$backend" <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

import numpy as np
import torch
import triton

backend = sys.argv[1]
assert platform.python_version() == '3.12.14'
assert np.__version__ == '2.2.6'
assert triton.__version__ == '3.5.1'
runtime = 'cu128' if backend == 'nvidia' else 'rocm6.4'
assert torch.__version__ == f'2.9.1+{runtime}', torch.__version__
assert bool(torch.version.hip) == (backend == 'amd')
assert torch.cuda.is_available(), 'No allocated GPU accessible; check the job and device visibility.'
assert (torch.ones(1, device='cuda') + 1).item() == 2
info = dict(python=platform.python_version(), torch=torch.__version__, torch_git=torch.version.git_version,
            triton=triton.__version__, cuda=torch.version.cuda, hip=torch.version.hip,
            gpu=torch.cuda.get_device_name(), system=platform.platform(),
            packages={d.metadata['Name']: d.version for d in metadata.distributions(
                path=[sysconfig.get_path('purelib')])},
            visibility={k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                   'ROCR_VISIBLE_DEVICES') if k in os.environ})
info['host_compiler'] = subprocess.check_output([os.environ.get('CC', 'cc'), '--version'], text=True).splitlines()[0]
Path(sys.prefix, 'environment.json').write_text(json.dumps(info, indent=2) + '\n')
print(json.dumps(info, indent=2))
PY
timeout 120s "$python" "$root/simple_divergence.py"
"$uv_tool" pip freeze --python "$python" > "$target/installed.txt"
echo "Package, GPU calculation, and small Triton probe passed: $target"
echo "Next: $python $root/reproduce_examples.py"
