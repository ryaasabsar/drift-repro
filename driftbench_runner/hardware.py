"""Hardware discovery that also works in CPU-only benchmark clients."""
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


SUPPORTED = {"nvidia": ("vllm", "sglang"),
             "amd": ("vllm", "sglang"), "tenstorrent": ("vllm",)}


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
    packages = {}
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
        "VLLM_BATCH_INVARIANT") if k in os.environ}
    if os.environ.get("CUDA_HOME"):
        result["cuda_compiler"] = command_output([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"])
    result["source_revisions"] = {}
    for key in ("TT_METAL_HOME", "VLLM_ROOT", "SGLANG_ROOT"):
        if os.environ.get(key):
            result["source_revisions"][key] = {
                "commit": command_output(["git", "-C", os.environ[key], "rev-parse", "HEAD"]),
                "changes": command_output(["git", "-C", os.environ[key], "status", "--porcelain", "--untracked-files=no"])}
    try:
        import torch
        result["cuda_runtime"] = torch.version.cuda
        result["rocm_runtime"] = getattr(torch.version, "hip", None)
        result["accelerators"] = [
            {"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)),
             "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]
    except ImportError:
        result["accelerators"] = []
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
        raise RuntimeError(f"No usable {vendor} accelerator found in this serving environment")


def validate_target(config):
    if config.get("transport") != "http":
        raise ValueError("Production inference requires transport=http; offline experiments are archived")
    vendor = config.get("hardware", {}).get("vendor", "nvidia")
    backend = config["backend"]
    if vendor not in SUPPORTED or backend not in SUPPORTED[vendor]:
        raise ValueError(f"Unsupported vendor/framework combination: {vendor}/{backend}")
