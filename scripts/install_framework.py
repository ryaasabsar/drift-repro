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

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = {
    "sglang": (".venv-sglang", "12.8.1", "sglang", "0.5.10.post1", "12.8"),
    "tensorrt": (".venv-trt", "13.0.2", "tensorrt-llm", "1.2.1", "13.0"),
}


def target_path(framework, requested):
    target = ROOT / (requested or FRAMEWORKS[framework][0])
    # Never let sync target the client/vLLM environment or a system interpreter.
    if (target.is_symlink() or target.resolve().parent != ROOT
            or not target.name.startswith(FRAMEWORKS[framework][0])):
        raise ValueError(f"Use a workspace directory named {FRAMEWORKS[framework][0]} or starting with it")
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


def verify(framework, target, env, dry_run=False):
    _, cuda, package, version, runtime = FRAMEWORKS[framework]
    python = target / "bin/python"
    run([ROOT / ".tools/uv", "pip", "check", "--python", python], env, dry_run)
    # Use the exact same library/compiler environment as the benchmark launcher.
    code = """
import importlib.metadata as metadata
import os
import subprocess
import sys
from driftbench_runner.serving import configure_serving_environment
package, expected, cuda, runtime = sys.argv[1:]
assert sys.version_info[:2] == (3, 12), 'Expected Python 3.12'
assert metadata.version(package) == expected, 'Framework version differs from snapshot'
configure_serving_environment({'backend': package, 'hardware': {'vendor': 'nvidia'},
    'launch': {'workspace_gcc': True, 'cuda_version': cuda}})
subprocess.run([os.environ['CUDA_HOME'] + '/bin/nvcc', '--version'], check=True)
subprocess.run([os.environ['CXX'], '--version'], check=True)
subprocess.run([sys.executable, '-c',
    'import torch; assert torch.version.cuda == ' + repr(runtime) +
    '; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'], check=True)
module = 'sglang.launch_server' if package == 'sglang' else 'tensorrt_llm.commands.serve'
subprocess.run([sys.executable, '-m', module, '--help'], check=True,
    stdout=subprocess.DEVNULL, timeout=120)
"""
    run([python, "-c", code, package, version, cuda, runtime], env, dry_run, timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=FRAMEWORKS)
    parser.add_argument("--venv", help="Workspace environment directory; defaults match the NVIDIA suites")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating files or downloading")
    parser.add_argument("--check", action="store_true", help="Check an existing installation without installing packages")
    args = parser.parse_args()
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("These pinned NVIDIA environments require Linux x86_64; they are not ROCm installers")
        target = target_path(args.framework, args.venv)
        _, cuda, _, _, _ = FRAMEWORKS[args.framework]
        env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".cache/uv"),
                   UV_PYTHON_INSTALL_DIR=str(ROOT / ".tools/python"),
                   XDG_CACHE_HOME=str(ROOT / ".cache"), PYTHONPATH=str(ROOT))
        if args.check:
            if not args.dry_run and not (target / "bin/python").exists():
                raise ValueError(f"Missing {target}/bin/python; run the installer without --check first")
            verify(args.framework, target, env, args.dry_run)
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
                if not target.exists():
                    run([uv, "venv", "--python", "3.12", target], env, args.dry_run)
                else:
                    run([python, "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Expected Python 3.12'"], env, args.dry_run)
                command = [uv, "pip", "sync", "--python", python,
                           ROOT / f"requirements.{args.framework}.lock.txt"]
                if args.framework == "tensorrt":
                    command += ["--index", "https://pypi.nvidia.com"]
                run(command, env, args.dry_run)
                run([uv, "pip", "install", "--python", python, "--no-deps", "-e", ROOT], env, args.dry_run)
                run([python, ROOT / "scripts/install_workspace_gcc.py"], env, args.dry_run)
                if not (ROOT / f".tools/cuda-{cuda}/bin/nvcc").is_file():
                    run([python, ROOT / "scripts/install_cuda_compiler.py", "--version", cuda], env, args.dry_run)
                verify(args.framework, target, env, args.dry_run)
            finally:
                if lock:
                    lock.close()
        print("Plan complete." if args.dry_run else
              f"Installation checks passed: {target}\nNext: run the standalone inference smoke suite on the target GPU.")
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
