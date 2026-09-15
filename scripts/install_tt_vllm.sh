#!/usr/bin/env bash
# Install the Python 3.12 TT vLLM environment. No driver changes or model downloads.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$root/.venv-tt-vllm"
plugin_commit=6d3bb2854f5f8885acc1b12111d929bffdebc36e
metal_commit=9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9
plugin="$root/.tools/vllm-tt-plugin-$plugin_commit"
metal="$root/.tools/tt-metal-$metal_commit"
# From this TT-Metal commit's tt_metal/sfpi-version (Debian x86_64 archive).
sfpi_version=7.69.0
sfpi_hash=b0c93362de2f69b0e4c335abd126ff59748f5dae316980d17aa1525ed523eb14
sfpi="$root/.tools/sfpi-$sfpi_version"
uv="$root/.tools/uv"
dry_run=false
check_only=false
toolchain_only=false
recreate=false

usage() {
    cat <<'EOF'
Usage: bash scripts/install_tt_vllm.sh [--dry-run | --check] [--recreate | --toolchain-only]

Install into .venv-tt-vllm with workspace-managed Python 3.12:
  vLLM 0.25.1 (empty/source build), vllm-tt-plugin compat/vllm-0.25.1,
  TTNN/TT-Metal 0.77.0, CPU PyTorch 2.11.0 and torchvision 0.26.0.

--dry-run   Print the installation plan; create/download nothing.
--check     Check the existing installation without installing or opening devices.
--recreate  Replace ONLY .venv-tt-vllm (e.g. an earlier Python 3.10 installation).
--toolchain-only  Install/check SFPI 7.69.0 for an existing TTNN 0.77.0 environment.
                  Leaves Python packages intact; no sudo or device access.

This is a pinned package baseline, not a validated P150b model deployment.
TT-Metal 0.77.0 documents Blackhole KMD >=2.8.0 and firmware 19.8.1.
The installer does not update the host driver, firmware or hugepages.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=true ;;
        --check) check_only=true ;;
        --toolchain-only) toolchain_only=true ;;
        --recreate) recreate=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done
if $recreate && { $check_only || $toolchain_only; }; then
    echo '--recreate cannot be combined with --check or --toolchain-only.' >&2
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

setup_sfpi() (
    # TT-Metal selects <runtime root>/runtime/sfpi before /opt/tenstorrent/sfpi.
    # Register both the installed wheel and the matching source runtime roots.
    if $dry_run; then
        echo "+ Install/check SFPI $sfpi_version (SHA256 $sfpi_hash) in $sfpi"
        echo '+ Register SFPI under the TTNN wheel and TT-Metal runtime directories; compile a Blackhole probe.'
        return
    fi
    local wheel compiler temporary
    wheel=$("$target/bin/python" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
d = distribution("ttnn")
assert d.version == "0.77.0", "SFPI pin requires TTNN 0.77.0"
p = Path(d.locate_file("ttnn")).resolve()
assert (p / "__init__.py").is_file(), f"Missing TTNN wheel: {p}"
required = [
    "toolchain/blackhole/firmware_ierisc.ld",
    "toolchain/blackhole/firmware_main_aerisc.ld",
    "toolchain/blackhole/firmware_brisc.ld",
    "lib/blackhole/tmu-crt0.o",
    "lib/blackhole/noc.o",
    "lib/blackhole/substitutes.o",
]
missing = [str(p / "runtime/hw" / name) for name in required
           if not (p / "runtime/hw" / name).is_file()]
if missing:
    raise RuntimeError("TTNN wheel is missing Blackhole runtime assets; repair the TTNN 0.77.0 installation:\n"
                       + "\n".join(missing))
print(p)
PY
    )
    if [[ -L "$sfpi" ]]; then
        echo "Refusing SFPI installation symlink: $sfpi" >&2; exit 1
    fi
    if [[ ! -e "$sfpi" ]] && ! $check_only; then
        temporary=$(mktemp -d "$root/.tools/.sfpi-install.XXXXXX")
        trap 'rm -rf -- "$temporary"' EXIT
        curl --fail --location --retry 3 \
            "https://github.com/tenstorrent/sfpi/releases/download/$sfpi_version/sfpi_${sfpi_version}_x86_64_debian.txz" \
            --output "$temporary/sfpi.txz"
        if ! printf '%s  %s\n' "$sfpi_hash" "$temporary/sfpi.txz" | sha256sum --check --status; then
            echo 'SFPI archive checksum mismatch; refusing to extract it.' >&2
            exit 1
        fi
        tar -xJf "$temporary/sfpi.txz" -C "$temporary" --no-same-owner
        printf '%s\n' "$sfpi_hash" > "$temporary/sfpi/.driftbench-archive-sha256"
        mv "$temporary/sfpi" "$sfpi"
    fi
    if [[ ! -f "$sfpi/.driftbench-archive-sha256" ]] || \
        [[ "$(cat "$sfpi/.driftbench-archive-sha256")" != "$sfpi_hash" ]]; then
        echo "Missing or unrecognized SFPI installation: $sfpi; use --toolchain-only to install it." >&2
        exit 1
    fi
    compiler="$sfpi/compiler/bin/riscv-tt-elf-g++"
    "$compiler" --version
    printf 'int sfpi_probe() { return 0; }\n' | "$compiler" \
        -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit \
        -mcpu=tt-bh -x c++ -c -o /dev/null -
    "$target/bin/python" - "$sfpi" "$wheel" "$metal" "$check_only" <<'PY'
from pathlib import Path
import sys
sfpi = Path(sys.argv[1]).resolve()
for root in map(Path, sys.argv[2:4]):
    if not root.is_dir():
        raise RuntimeError(f"Missing runtime root: {root}; run the full installer first")
    link = root / "runtime" / "sfpi"
    if link.is_symlink() and link.resolve() == sfpi:
        continue
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"Existing SFPI at {link}; refusing to replace it")
    if sys.argv[4] == "true":
        raise RuntimeError(f"Missing {link}; run installer --toolchain-only")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(sfpi, target_is_directory=True)
    print(f"Registered {link} -> {sfpi}")
print("SFPI Blackhole compilation probe passed (no device opened).")
PY
)

write_activation() {
    if $dry_run || $check_only; then return; fi
    {
        printf 'source %q\n' "$target/bin/activate"
        printf 'export TT_METAL_HOME=%q\n' "$metal"
        # A source checkout provides models.*, not the built firmware runtime.
        # With no override, TTNN selects its wheel's bundled runtime assets.
        printf 'unset TT_METAL_RUNTIME_ROOT\n'
        printf 'export DRIFTBENCH_PYTHON=%q\n' "$root/.venv-client/bin/python"
        printf 'unset VLLM_TARGET_DEVICE\n'
    } > "$target/activate-tt.sh"
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
unset TT_METAL_RUNTIME_ROOT
export PYTHONPATH="$root"

echo 'TT environment: Python 3.12, vLLM 0.25.1, TTNN 0.77.0, CPU PyTorch 2.11.0.'
echo 'Package setup only: TT-Metal 0.77.0 documents Blackhole KMD >=2.8.0 / firmware 19.8.1.'

if $toolchain_only && ! $dry_run && [[ ! -x "$target/bin/python" ]]; then
    echo 'Missing TT environment; run the full installer first.' >&2
    exit 1
fi
if ! $check_only; then
    if ! $dry_run; then
        mkdir -p "$root/.tools"
        # Serialize with the other installers that share workspace uv caches.
        exec 9>"$root/.tools/framework-install.lock"
        flock 9
    fi
fi
if $toolchain_only; then
    setup_sfpi
    write_activation
    if $dry_run; then
        echo 'Compiler repair plan complete; nothing changed.'
    else
        echo 'Compiler check passed. Python packages and host drivers were not changed.'
        printf 'Activate the corrected runtime: source %q\n' "$target/activate-tt.sh"
    fi
    exit 0
fi
if ! $check_only; then
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
    setup_sfpi

    if ! $dry_run; then
        # The wheel supplies TTNN; the matching checkout supplies models.*.
        # A .pth appends the source root so it cannot shadow the installed wheel.
        "$target/bin/python" - "$metal" <<'PY'
from pathlib import Path
import sys
import sysconfig
Path(sysconfig.get_path("purelib"), "driftbench_tt_models.pth").write_text(sys.argv[1] + "\n")
PY
        write_activation
    fi
fi

if $dry_run; then
    echo 'Plan complete. Final checks: SFPI compilation, package dependencies, CPU torch, TTNN import, TT plugin discovery.'
    exit 0
fi
if [[ ! -x "$target/bin/python" ]]; then
    echo 'Missing .venv-tt-vllm/bin/python; run without --check first.' >&2
    exit 1
fi
if $check_only; then setup_sfpi; fi
unset VLLM_TARGET_DEVICE
"$target/bin/python" - "$target" "$plugin_commit" "$metal_commit" "$sfpi_version" "$sfpi_hash" <<'PY'
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
          "tt_metal_commit": sys.argv[3], "sfpi_version": sys.argv[4],
          "sfpi_archive_sha256": sys.argv[5],
          "checks": "package/import checks and SFPI Blackhole compilation probe",
          "hardware_validated": False}
(target / "tt-install-report.json").write_text(json.dumps(report, indent=2) + "\n")
PY
echo 'Package installation checked. P150b inference has NOT been validated.'
echo 'TT-Metal 0.77.0 requires Blackhole KMD >=2.8.0 and firmware 19.8.1 per its release instructions.'
printf 'Activate: source %q\n' "$target/activate-tt.sh"
