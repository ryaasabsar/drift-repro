#!/usr/bin/env bash
# Install the Python 3.12 TT vLLM environment. No driver changes or model downloads.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$root/.venv-tt-vllm"
plugin_commit=6d3bb2854f5f8885acc1b12111d929bffdebc36e
metal_commit=9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
plugin="$root/.tools/vllm-tt-plugin-$plugin_commit"
metal="$root/.tools/tt-metal-$metal_commit"
uv="$root/.tools/uv"
dry_run=false
check_only=false
recreate=false

usage() {
    cat <<'EOF'
Usage: bash scripts/install_tt_vllm.sh [--dry-run | --check] [--recreate]

Install into .venv-tt-vllm with workspace-managed Python 3.12:
  vLLM 0.25.1 (empty/source build), vllm-tt-plugin compat/vllm-0.25.1,
  TTNN/TT-Metal 0.77.0, CPU PyTorch 2.11.0 and torchvision 0.26.0.

--dry-run   Print the installation plan; create/download nothing.
--check     Check the existing installation without installing or opening devices.
--recreate  Replace ONLY .venv-tt-vllm (e.g. an earlier Python 3.10 installation).

This is a pinned package baseline, not a validated P150b model deployment.
TT-Metal 0.77.0 documents Blackhole KMD >=2.8.0 and firmware 19.8.1.
The installer does not update the host driver, firmware or hugepages.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --check) check_only=true ;;
        --recreate) recreate=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done
if $check_only && $recreate; then
    echo '--check and --recreate cannot be combined.' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This installer requires Linux x86_64 (TTNN manylinux wheel).' >&2
    exit 1
fi
if [[ -L "$target" || ( -e "$target" && ! -f "$target/pyvenv.cfg" ) ]]; then
    echo "Refusing to modify symlink/non-venv target: $target" >&2
    exit 1
fi

run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if ! $dry_run; then "$@"; fi
}

checkout() {
    local url="$1" dest="$2" commit="$3"
    if [[ -L "$dest" ]]; then
        echo "Refusing source symlink: $dest" >&2; exit 1
    fi
    if [[ ! -d "$dest" ]]; then
        run git init "$dest"
        run git -C "$dest" remote add origin "$url"
        run git -C "$dest" fetch --depth 1 origin "$commit"
        run git -C "$dest" checkout --detach FETCH_HEAD
    elif [[ "$(git -C "$dest" rev-parse HEAD)" != "$commit" || -n "$(git -C "$dest" status --porcelain --untracked-files=no)" ]]; then
        echo "Source checkout differs from its pin; refusing to overwrite: $dest" >&2
        exit 1
    fi
}

# Override an activated CUDA/ROCm environment when invoking uv and subprocesses.
export UV_CACHE_DIR="$root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$root/.tools/python"
export XDG_CACHE_HOME="$root/.cache"
export VIRTUAL_ENV="$target"
export UV_PYTHON="$target/bin/python"
export UV_TORCH_BACKEND=cpu
export PATH="$target/bin:$root/.tools:$PATH"
export VLLM_TARGET_DEVICE=empty
export TT_METAL_HOME="$metal"
export TT_METAL_RUNTIME_ROOT="$metal"
export PYTHONPATH="$root"

echo 'TT environment: Python 3.12, vLLM 0.25.1, TTNN 0.77.0, CPU PyTorch 2.11.0.'
echo 'Package setup only: TT-Metal 0.77.0 documents Blackhole KMD >=2.8.0 / firmware 19.8.1.'

if ! $check_only; then
    if ! $dry_run; then
        mkdir -p "$root/.tools"
        # Serialize with the other installers that share workspace uv caches.
        exec 9>"$root/.tools/framework-install.lock"
        flock 9
    fi
    run bash "$root/scripts/ensure_uv.sh"
    if [[ ! -e "$target" ]] || $recreate; then
        creation=("$uv" venv --managed-python --python 3.12)
        if $recreate; then creation+=(--clear); fi
        run "${creation[@]}" "$target"
    else
        run "$target/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing TT environment is not Python 3.12; rerun with --recreate"'
    fi
    checkout https://github.com/tenstorrent/vllm-tt-plugin.git "$plugin" "$plugin_commit"
    checkout https://github.com/tenstorrent/tt-metal.git "$metal" "$metal_commit"
    run git -C "$metal" submodule update --init --recursive --depth 1

    if ! $dry_run; then
        cat > "$target/tt-constraints.txt" <<'EOF'
torch==2.11.0+cpu
torchvision==0.26.0+cpu
ttnn==0.77.0
transformers==5.12.1
numpy==1.26.4
EOF
    fi
    export UV_CONSTRAINT="$target/tt-constraints.txt"
    run "$uv" pip install --python "$target/bin/python" \
        'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' 'ttnn==0.77.0' \
        'transformers==5.12.1' 'numpy==1.26.4' 'pytest>=8,<9' 'accelerate==1.7.0'
    # An old tt-vllm-plugin installation registers conflicting entry points.
    run "$uv" pip uninstall --python "$target/bin/python" tt-vllm-plugin
    # Force a source rebuild even if a PyPI CUDA wheel with this version exists.
    run "$uv" pip uninstall --python "$target/bin/python" vllm
    run bash -euo pipefail -c 'cd "$1"; source docs/install-vllm-tt.sh' _ "$plugin"
    run "$uv" pip install --python "$target/bin/python" --no-deps -e "$root"

    if ! $dry_run; then
        # The wheel supplies TTNN; the matching checkout supplies models.*.
        # A .pth appends the source root so it cannot shadow the installed wheel.
        "$target/bin/python" - "$metal" <<'PY'
from pathlib import Path
import sys
import sysconfig
Path(sysconfig.get_path("purelib"), "driftbench_tt_models.pth").write_text(sys.argv[1] + "\n")
PY
        {
            printf 'source %q\n' "$target/bin/activate"
            printf 'export TT_METAL_HOME=%q\n' "$metal"
            printf 'export TT_METAL_RUNTIME_ROOT=%q\n' "$metal"
            printf 'export DRIFTBENCH_PYTHON=%q\n' "$target/bin/python"
            printf 'unset VLLM_TARGET_DEVICE\n'
        } > "$target/activate-tt.sh"
    fi
fi

if $dry_run; then
    echo 'Plan complete. Final checks: package dependencies, CPU torch, TTNN import, TT plugin discovery.'
    exit 0
fi
if [[ ! -x "$target/bin/python" ]]; then
    echo 'Missing .venv-tt-vllm/bin/python; run without --check first.' >&2
    exit 1
fi
unset VLLM_TARGET_DEVICE
"$target/bin/python" - "$target" "$plugin_commit" "$metal_commit" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

target = Path(sys.argv[1])
assert sys.version_info[:2] == (3, 12), "Expected Python 3.12"
expected = {"vllm": "0.25.1", "vllm-tt-plugin": "0.1.0", "ttnn": "0.77.0",
            "torch": "2.11.0+cpu", "torchvision": "0.26.0+cpu", "transformers": "5.12.1"}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise RuntimeError(f"Expected {name} {version}, found {actual}")
    print(name, actual, flush=True)

# Upstream explicitly overrides OpenCV's numpy>=2 dependency. Check every other
# active requirement, rather than ignoring pip-check failures wholesale.
errors = []
packages = {canonicalize_name(d.metadata["Name"]): d.version for d in metadata.distributions()}
for dist in metadata.distributions():
    name = canonicalize_name(dist.metadata["Name"])
    if name.startswith("nvidia-") or name == "tt-vllm-plugin":
        errors.append(f"Unexpected CUDA/legacy package: {name}")
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        dep = canonicalize_name(req.name)
        installed = packages.get(dep)
        if installed is None or (req.specifier and not req.specifier.contains(installed, prereleases=True)):
            if name == "vllm" and dep == "opencv-python-headless" and installed == "4.11.0.86":
                print("Using upstream's documented OpenCV 4.11 / numpy<2 override.", flush=True)
                continue
            errors.append(f"{name} requires {req}; installed: {installed}")
if errors:
    raise RuntimeError("\n".join(errors))

import torch
import torchvision
import ttnn
import vllm
import vllm_tt_plugin
from vllm_tt_plugin.entrypoints import platform_plugin
assert torch.version.cuda is None and torch.version.hip is None, "Expected CPU PyTorch"
plugins = {p.name: p.value for p in metadata.entry_points(group="vllm.platform_plugins")}
assert plugins.get("tt") == "vllm_tt_plugin.entrypoints:platform_plugin", plugins
assert platform_plugin(), "TT platform discovery failed"
for label, commit in (("vllm-tt-plugin", sys.argv[2]), ("tt-metal", sys.argv[3])):
    source = target.parent / ".tools" / f"{label}-{commit}"
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != commit or changes:
        raise RuntimeError(f"{label} source differs from the installation pin: {source}")
plugin_source = target.parent / ".tools" / f"vllm-tt-plugin-{sys.argv[2]}"
if not Path(vllm_tt_plugin.__file__).resolve().is_relative_to(plugin_source):
    raise RuntimeError("Imported TT plugin is outside the pinned source checkout")
print("CPU torch, TTNN, vLLM and TT plugin imports passed.", flush=True)
report = {"python": sys.version, "packages": packages, "plugin_commit": sys.argv[2],
          "tt_metal_commit": sys.argv[3], "checks": "package/import checks only",
          "hardware_validated": False}
(target / "tt-install-report.json").write_text(json.dumps(report, indent=2) + "\n")
PY
echo 'Package installation checked. P150b inference has NOT been validated.'
echo 'TT-Metal 0.77.0 requires Blackhole KMD >=2.8.0 and firmware 19.8.1 per its release instructions.'
printf 'Activate: source %q\n' "$target/activate-tt.sh"
