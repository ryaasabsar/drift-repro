#!/usr/bin/env python3
"""Install the C++20 toolchain used on the local Ubuntu 20.04 host."""
import io
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
binary = ROOT / ".tools/micromamba"
if not binary.exists():
    with urllib.request.urlopen("https://micro.mamba.pm/api/micromamba/linux-64/2.8.1", timeout=60) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data)) as package:
        data = package.extractfile("bin/micromamba").read()
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(data)
    binary.chmod(0o755)
lock = ROOT / "requirements.gcc13.explicit.txt"
packages = ["--file", str(lock)] if lock.exists() else ["-c", "conda-forge", "gxx_linux-64=13"]
subprocess.run([str(binary), "create", "-y", "-r", str(ROOT / ".tools/mamba-root"),
                "-p", str(ROOT / ".tools/gcc13"), *packages], check=True)
