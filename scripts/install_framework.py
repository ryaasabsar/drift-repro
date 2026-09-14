#!/usr/bin/env python3
"""Restore the pinned Linux x86_64 NVIDIA serving environments without sudo."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]


class Framework(NamedTuple):
    venv: str
    cuda: str
    package: str
    version: str
    runtime: str


FRAMEWORKS = {
    "sglang": Framework(".venv-sglang", "12.8.1", "sglang", "0.5.10.post1", "12.8"),
    "tensorrt": Framework(".venv-trt", "12.8.1", "tensorrt-llm", "0.20.0", "12.8"),
}


def target_path(framework, requested):
    default = FRAMEWORKS[framework].venv
    target = ROOT / (requested or default)
    # Never let sync target the client/vLLM environment or a system interpreter.
    if (target.is_symlink() or target.resolve().parent != ROOT
            or not target.name.startswith(default)):
        raise ValueError(f"Use a workspace directory named {default} or starting with it")
    if target.exists() and not (target / "pyvenv.cfg").is_file():
        raise ValueError(f"Existing target is not a virtual environment: {target}")
    return target.resolve()


def run(command, env, dry_run=False, timeout=None):
    command = [str(part) for part in command]
    display = command[:]
    if "-c" in display:
        display[display.index("-c") + 1] = "<Python installation check>"
    print("+ " + shlex.join(display), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=timeout)


def uses_managed_python(target):
    config = dict(line.split(" = ", 1) for line in (target / "pyvenv.cfg").read_text().splitlines() if " = " in line)
    home = Path(config.get("home", "/")).resolve()
    return (ROOT / ".tools/python").resolve() in home.parents


def verify(framework, target, env, dry_run=False, check_gpu=False):
    spec = FRAMEWORKS[framework]
    python = target / "bin/python"
    run([ROOT / ".tools/uv", "pip", "check", "--python", python], env, dry_run)
    command = [python, "-m", "driftbench_runner.framework_check",
               spec.package, spec.version, spec.cuda, spec.runtime]
    if check_gpu:
        command.append("--check-gpu")
    run(command, env, dry_run, timeout=450)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("--venv", help="Workspace environment directory; defaults match the NVIDIA suites")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating files or downloading")
    parser.add_argument("--check", action="store_true", help="Check an existing installation without installing packages")
    parser.add_argument("--check-gpu", action="store_true", help="Check an existing installation and run CUDA/Triton probes on an allocated GPU")
    args = parser.parse_args()
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("These pinned NVIDIA environments require Linux x86_64; they are not ROCm installers")
        target = target_path(args.framework, args.venv)
        cuda = FRAMEWORKS[args.framework].cuda
        env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                   UV_PYTHON_INSTALL_DIR=str(ROOT / ".tools/python"),
                   XDG_CACHE_HOME=str(ROOT / ".cache"), PYTHONPATH=str(ROOT))
        if args.check or args.check_gpu:
            if not args.dry_run and not (target / "bin/python").exists():
                raise ValueError(f"Missing {target}/bin/python; run the installer without --check first")
            verify(args.framework, target, env, args.dry_run, args.check_gpu)
        else:
            # Both installers share GCC/uv/CUDA caches. Serialize actual installs.
            lock = None
            if not args.dry_run:
                (ROOT / ".tools").mkdir(exist_ok=True)
                lock = (ROOT / ".tools/framework-install.lock").open("w")
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                run(["bash", ROOT / "scripts/ensure_uv.sh"], env, args.dry_run)
                uv = ROOT / ".tools/uv"
                python = target / "bin/python"
                if not target.exists() or not uses_managed_python(target):
                    if target.exists():
                        print(f"Replacing system-Python environment at {target} with workspace-managed Python and complete headers.", flush=True)
                    run([uv, "venv", "--managed-python", "--python", "3.12", "--clear", target], env, args.dry_run)
                else:
                    run([python, "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Expected Python 3.12'"], env, args.dry_run)
                # TensorRT 0.20 requires a source build of its removed xgrammar
                # wheel. Prepare the workspace compiler before syncing packages.
                run([python, ROOT / "scripts/install_workspace_gcc.py"], env, args.dry_run)
                if args.framework == "tensorrt":
                    env.update(CC=str(ROOT / ".tools/gcc13/bin/x86_64-conda-linux-gnu-gcc"),
                               CXX=str(ROOT / ".tools/gcc13/bin/x86_64-conda-linux-gnu-g++"),
                               CMAKE_BUILD_PARALLEL_LEVEL="2")
                command = [uv, "pip", "sync", "--python", python,
                           ROOT / f"requirements.{args.framework}.lock.txt"]
                if args.framework == "tensorrt":
                    command += ["--index", "https://pypi.org/simple", "--default-index", "https://pypi.nvidia.com"]
                run(command, env, args.dry_run)
                run([uv, "pip", "install", "--python", python, "--no-deps", "-e", ROOT], env, args.dry_run)
                if not (ROOT / f".tools/cuda-{cuda}/bin/nvcc").is_file():
                    run([python, ROOT / "scripts/install_cuda_compiler.py", "--version", cuda], env, args.dry_run)
                verify(args.framework, target, env, args.dry_run)
            finally:
                if lock:
                    lock.close()
        print("Plan complete." if args.dry_run else
              f"Package/compiler checks passed: {target}\n" +
              ("CUDA/Triton probes passed; model inference still needs a smoke run." if args.check_gpu else
               "GPU execution was not tested. Run this script with --check-gpu on your allocated GPU, then run the model smoke suite."))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = f"command exited with status {exc.returncode}; see the output above"
        elif isinstance(exc, subprocess.TimeoutExpired):
            detail = "installation check timed out; inspect the output and rerun with --check"
        else:
            detail = str(exc)
        print(f"Installation failed: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
