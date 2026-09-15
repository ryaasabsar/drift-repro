"""Installation probes run by the serving environment's own Python interpreter."""
import argparse
import faulthandler
import importlib.metadata as metadata
import os
from pathlib import Path
import runpy
import subprocess
import sys
import sysconfig
import tempfile

from .serving import configure_serving_environment


def check_headers():
    # Match Triton's compiler/include path to catch incomplete Python headers.
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "python_headers.c"
        source.write_text("#include <Python.h>\nint check_headers(void) { return PY_MAJOR_VERSION; }\n")
        subprocess.run([os.environ["CC"], "-fPIC", "-c", str(source),
                        "-I" + sysconfig.get_path("include"), "-o", str(Path(tmp) / "headers.o")],
                       check=True, timeout=30)


def check_runtime(runtime, check_gpu):
    import torch

    if torch.version.cuda != runtime:
        raise ValueError(f"Expected CUDA {runtime}, found {torch.version.cuda}")
    print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, flush=True)
    if check_gpu:
        print("Testing CUDA calculation", flush=True)
        x = torch.ones(1, device="cuda")
        if (x + 1).item() != 2:
            raise RuntimeError("CUDA calculation failed")
        import triton

        triton.runtime.driver.active.get_current_device()
        print("CUDA and Triton initialization passed:", torch.cuda.get_device_name(0), flush=True)


def check_serving_import(package):
    module = "sglang.launch_server"
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        sys.argv = [module, "--help"]
        runpy.run_module(module, run_name="__main__")
    finally:
        faulthandler.cancel_dump_traceback_later()


def check_installation(args):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Expected Python 3.12")
    installed = metadata.version(args.package)
    if installed != args.version:
        raise ValueError(f"Expected {args.package} {args.version}, found {installed}")
    configure_serving_environment({"backend": args.package, "hardware": {"vendor": "nvidia"},
                                   "launch": {"workspace_gcc": True, "cuda_version": args.cuda}})
    subprocess.run([os.environ["CUDA_HOME"] + "/bin/nvcc", "--version"], check=True)
    subprocess.run([os.environ["CXX"], "--version"], check=True)
    check_headers()
    # Start fresh processes after configuring LD_LIBRARY_PATH: native serving
    # libraries need it at exec time.
    command = [sys.executable, "-u", "-m", "driftbench_runner.framework_check",
               args.package, args.version, args.cuda, args.runtime]
    gpu_args = ["--check-gpu"] if args.check_gpu else []
    subprocess.run(command + ["--phase", "runtime"] + gpu_args, check=True, timeout=90)
    log = Path(os.environ["XDG_CACHE_HOME"]) / (args.package + "-install-check.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print("Serving import log:", log, flush=True)
    with log.open("w") as output:
        subprocess.run(command + ["--phase", "serving"], check=True,
                       stdout=output, stderr=subprocess.STDOUT, timeout=300)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("sglang",))
    parser.add_argument("version")
    parser.add_argument("cuda")
    parser.add_argument("runtime")
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--phase", choices=("all", "runtime", "serving"), default="all")
    args = parser.parse_args()
    if args.phase == "runtime":
        check_runtime(args.runtime, args.check_gpu)
    elif args.phase == "serving":
        check_serving_import(args.package)
    else:
        check_installation(args)


if __name__ == "__main__":
    main()
