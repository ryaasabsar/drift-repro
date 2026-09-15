"""Hardware discovery that also works in CPU-only benchmark clients."""
import importlib.metadata
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED = {"nvidia": ("vllm", "sglang"),
             "amd": ("vllm", "sglang"), "tenstorrent": ("vllm",)}


def probe_accelerators(timeout=30, attempts=2):
    result = {}
    for attempt in range(attempts):
        try:
            process = subprocess.run([sys.executable, '-m', 'driftbench_runner.accelerator_probe'],
                                     capture_output=True, text=True, timeout=timeout)
            result = json.loads(process.stdout) if process.returncode == 0 else {
                'accelerator_error': f'Probe exited {process.returncode}: {process.stderr[-2000:]}'}
        except subprocess.TimeoutExpired:
            result = {'accelerator_error': f'CUDA/HIP initialization timed out after {timeout}s'}
        except (OSError, ValueError) as exc:
            result = {'accelerator_error': str(exc)}
        if not result.get('accelerator_error'):
            return result
        if attempt + 1 < attempts:
            from .logging import event
            event('Accelerator probe failed; retrying in a fresh process', stage='hardware',
                  level='warning', error=result['accelerator_error'])
    return {'accelerators': [], **result}


def hardware_guidance(metadata):
    env = metadata.get('runtime_environment', {})
    guidance = []
    runtime = metadata.get('cuda_runtime') or ''
    smi = metadata.get('nvidia', {}).get('stdout', '')
    if runtime.startswith('13.') and '570.' in smi:
        guidance.append('CUDA 13 wheels do not match the R570 driver baseline. Restore the pinned CUDA 12.8 environment; installing nvcc alone cannot change the PyTorch runtime.')
    if env.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        guidance.append('CUDA_VISIBLE_DEVICES hides all GPUs. Request a GPU allocation and preserve the scheduler-provided device mask.')
    if any('/stubs' in p for p in env.get('LD_LIBRARY_PATH', '').split(':')):
        guidance.append('LD_LIBRARY_PATH contains CUDA stub libraries; use the real host driver libcuda.so at runtime.')
    guidance.append('Run doctor inside the active GPU job/container, using the serving Python. Preserve CUDA_VISIBLE_DEVICES; do not replace a scheduler allocation with a physical index.')
    guidance.append('If the pinned CUDA 12.8 calculation still fails, request another healthy allocation or send startup-diagnostics.json to the administrator. A node/driver fault cannot be repaired by changing sampling settings.')
    return guidance


def command_output(command):
    if not shutil.which(command[0]):
        return {"available": False}
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"available": True, "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": str(exc)}


def discover():
    packages = {d.metadata['Name'].lower().replace('_', '-'): d.version
                for d in importlib.metadata.distributions() if d.metadata['Name']}
    package_sources = {}
    for name in ("torch", "vllm", "sglang", "transformers", "tokenizers",
                 "triton", "pytorch-triton-rocm", "ttnn", "tt-metal", "tt-smi",
                 "vllm-tt-plugin"):
        try:
            packages[name] = importlib.metadata.version(name)
            raw_source = importlib.metadata.distribution(name).read_text("direct_url.json")
            if raw_source:
                source = json.loads(raw_source)
                # Record VCS commit metadata without URLs that might contain credentials.
                package_sources[name] = {k: source[k] for k in ("vcs_info", "dir_info", "archive_info") if k in source}
        except importlib.metadata.PackageNotFoundError:
            pass
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
              "package_sources": package_sources,
              "nvidia": command_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader"]),
              "amd": command_output(["rocm-smi", "--showproductname", "--showuniqueid", "--showdriverversion", "--json"]),
              "amd_smi": command_output(["amd-smi", "static", "--json"])}
    result['driver_versions'] = {'nvidia': sorted({row[3].strip() for row in
        csv.reader(result['nvidia'].get('stdout', '').splitlines()) if len(row) == 5})}
    # PCI sysfs does not require CUDA, ROCm, or a working management CLI.
    devices = []
    for entry in sorted(Path("/sys/bus/pci/devices").glob("*")):
        try:
            vendor = (entry / "vendor").read_text().strip()
            if vendor in ("0x10de", "0x1002", "0x1e52"):
                devices.append({"address": entry.name, "vendor_id": vendor,
                                "device_id": (entry / "device").read_text().strip()})
        except OSError:
            continue
    result["pci_devices"] = devices
    result["tenstorrent_device_nodes"] = sorted(str(p) for p in Path("/dev/tenstorrent").glob("*"))
    result["runtime_environment"] = {k: os.environ[k] for k in (
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
        "VLLM_TARGET_DEVICE", "VLLM_USE_V1", "VLLM_ROCM_USE_AITER", "MESH_DEVICE", "ARCH_NAME", "TT_VISIBLE_DEVICES",
        "TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "CUDA_HOME", "ROCM_HOME", "TORCHDYNAMO_DISABLE",
        "CXX", "CC", "NVCC_CCBIN", "CPATH", "LD_LIBRARY_PATH", "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG", "SGLANG_ENABLE_DETERMINISTIC_INFERENCE",
        "VLLM_BATCH_INVARIANT", "PYTORCH_NVML_BASED_CUDA_CHECK",
        "NVIDIA_VISIBLE_DEVICES") if k in os.environ}
    if os.environ.get("CUDA_HOME"):
        result["cuda_compiler"] = command_output([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"])
    result["source_revisions"] = {}
    for key in ("TT_METAL_HOME", "VLLM_ROOT", "SGLANG_ROOT"):
        if os.environ.get(key):
            result["source_revisions"][key] = {
                "commit": command_output(["git", "-C", os.environ[key], "rev-parse", "HEAD"]),
                "changes": command_output(["git", "-C", os.environ[key], "status", "--porcelain", "--untracked-files=no"])}
    if 'torch' in packages:
        result.update(probe_accelerators())
    else:
        result['accelerators'] = []
    if result.get('accelerator_error'):
        result['guidance'] = hardware_guidance(result)
    return result


def require_hardware(vendor, metadata):
    if vendor not in SUPPORTED:
        raise ValueError(f"Unknown accelerator vendor: {vendor}")
    if vendor == "tenstorrent":
        present = bool(metadata["tenstorrent_device_nodes"] or
                       any(d["vendor_id"] == "0x1e52" for d in metadata["pci_devices"]))
    else:
        present = bool(metadata["accelerators"]) and bool(metadata.get("rocm_runtime")) == (vendor == "amd")
    if not present:
        detail = metadata.get('accelerator_error', 'No devices visible to this runtime')
        guidance = ' '.join(metadata.get('guidance', hardware_guidance(metadata)))
        raise RuntimeError(f"No usable {vendor} accelerator: {detail}. {guidance}")


def validate_target(config):
    if config.get("transport") != "http":
        raise ValueError("Production inference requires transport=http; offline experiments are archived")
    vendor = config.get("hardware", {}).get("vendor", "nvidia")
    backend = config["backend"]
    if vendor not in SUPPORTED or backend not in SUPPORTED[vendor]:
        raise ValueError(f"Unsupported vendor/framework combination: {vendor}/{backend}")
