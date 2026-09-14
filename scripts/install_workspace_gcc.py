#!/usr/bin/env python3
"""Install or reuse the pinned workspace C++20 toolchain without sudo."""
import io
from pathlib import Path
import subprocess
import platform
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This helper requires Linux x86_64")
    compiler = ROOT / ".tools/gcc13/bin/x86_64-conda-linux-gnu-g++"
    gcc = compiler.with_name("x86_64-conda-linux-gnu-gcc")
    if compiler.is_file() and gcc.is_file():
        for executable in (gcc, compiler):
            subprocess.run([str(executable), "--version"], check=True)
        return
    lock = ROOT / "requirements.gcc13.explicit.txt"
    if not lock.is_file():
        raise SystemExit(f"Missing pinned toolchain: {lock}")
    binary = ROOT / ".tools/micromamba"
    if not binary.exists():
        with urllib.request.urlopen("https://micro.mamba.pm/api/micromamba/linux-64/2.8.1", timeout=60) as response:
            data = response.read()
        with tarfile.open(fileobj=io.BytesIO(data)) as package:
            data = package.extractfile("bin/micromamba").read()
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(data)
        binary.chmod(0o755)
    subprocess.run([str(binary), "create", "-y", "-r", str(ROOT / ".tools/mamba-root"),
                    "-p", str(ROOT / ".tools/gcc13"), "--file", str(lock)], check=True)


if __name__ == "__main__":
    main()
