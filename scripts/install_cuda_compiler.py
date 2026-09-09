#!/usr/bin/env python3
"""Workspace-only CUDA compiler/headers, from NVIDIA's checksummed archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["12.8.1", "13.0.2"], required=True)
    args = parser.parse_args()
    if __import__("platform").machine() != "x86_64":
        raise SystemExit("This helper selects Linux x86_64 archives")
    with urllib.request.urlopen(BASE + f"redistrib_{args.version}.json", timeout=60) as response:
        manifest = json.load(response)
    target = ROOT / ".tools" / f"cuda-{args.version}"
    cache = ROOT / ".cache" / "cuda-archives"
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = target / "redistribution-lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    components = ["cuda_nvcc", "cuda_cudart", "cuda_cccl"]
    if args.version.startswith("13."):
        components += ["cuda_crt", "libnvvm"]
    for name in components:
        entry = manifest[name]["linux-x86_64"]
        if lock.get(name) == entry:
            continue
        archive = cache / Path(entry["relative_path"]).name
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != entry["sha256"]:
            print(f"Downloading {name} {manifest[name]['version']}", flush=True)
            urllib.request.urlretrieve(BASE + entry["relative_path"], archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            with tarfile.open(archive) as package:
                package.extractall(tmp, filter="data")
            roots = list(Path(tmp).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError("Unexpected archive layout")
            for item in roots[0].rglob("*"):
                dest = target / item.relative_to(roots[0])
                if item.is_symlink() and dest.is_symlink() and item.readlink() == dest.readlink():
                    dest.unlink()
            shutil.copytree(roots[0], target, dirs_exist_ok=True, symlinks=True)
        lock[name] = entry
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    # NVIDIA redistribution archives use lib; several JIT builders expect lib64.
    if (target / "lib").is_dir() and not (target / "lib64").exists():
        (target / "lib64").symlink_to("lib", target_is_directory=True)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"CUDA_HOME={target}", flush=True)


if __name__ == "__main__":
    main()
