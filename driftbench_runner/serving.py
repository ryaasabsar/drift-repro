"""Launch in the framework's own environment; capture serving-host provenance."""
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit

from .common import ROOT, digest, local_environment, now, read_json, write_json
from .hardware import discover, require_hardware, validate_target
from .http_backend import HTTPBackend
from .logging import activity, event


def serving_identity(config):
    return {**{k: config.get(k) for k in ("backend", "model", "revision", "seed", "hardware", "engine", "launch")},
            "served_model_name": config.get("server", {}).get("model_name", config["model"])}


def model_snapshot_alias(config):
    return ROOT / ".cache/serving-models" / config["revision"] / config["model"].split("/")[-1]


def launch_command(config, model_path=None):
    validate_target(config)
    if config.get("transport") != "http":
        raise ValueError("serve requires transport=http")
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Pin the model revision to a 40-character Hub commit before serving")
    url = urlsplit(config["server"]["base_url"])
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Launcher requires a local HTTP base_url; use the remote address on the benchmark client")
    launch = config.get("launch", {})
    backend = config["backend"]
    model = model_path or config["model"]
    alias = config["server"].get("model_name", config["model"])
    prefix = launch.get("command_prefix")
    if prefix is not None and (not isinstance(prefix, list) or not all(isinstance(x, str) for x in prefix)):
        raise ValueError("command_prefix must be an argv array (never a shell command)")
    if prefix and any(x.startswith("--") for x in prefix):
        raise ValueError("command_prefix selects the executable; put options in engine or extra_args")
    if backend == "vllm":
        command = list(prefix or [sys.executable, "-m", "vllm.entrypoints.openai.api_server"])
        command += ["--model", model, "--revision", config["revision"], "--tokenizer-revision", config["revision"],
                    "--served-model-name", alias, "--generation-config", "vllm", "--seed", str(config["seed"])]
        mapping = {k: "--" + k.replace("_", "-") for k in (
            "dtype", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization",
            "tensor_parallel_size", "enforce_eager", "enable_prefix_caching", "language_model_only",
            "attention_backend", "block_size", "additional_config", "max_num_partial_prefills")}
    elif backend == "sglang":
        command = list(prefix or [sys.executable, "-m", "sglang.launch_server"])
        command += ["--model-path", model, "--revision", config["revision"], "--served-model-name", alias,
                    "--random-seed", str(config["seed"])]
        mapping = {"max_model_len": "--context-length", "tensor_parallel_size": "--tp-size"}
        mapping.update({k: "--" + k.replace("_", "-") for k in (
            "dtype", "mem_fraction_static", "max_running_requests", "chunked_prefill_size", "attention_backend",
            "disable_radix_cache", "disable_cuda_graph", "disable_piecewise_cuda_graph", "disable_overlap_schedule", "max_mamba_cache_size",
            "page_size", "tt_visible_devices", "mesh_shape", "disable_custom_all_reduce",
            "enable_deterministic_inference", "sampling_defaults")})
    import json
    for key, value in config["engine"].items():
        if key not in mapping:
            raise ValueError(f"Unmapped {backend} engine option: {key}")
        flag = mapping[key]
        if flag is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif backend == "vllm":
                command.append("--no-" + flag[2:])
            # SGLang disable-* switches default to false.
        else:
            command += [flag, json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)]
    command += ["--host", launch.get("host", "127.0.0.1"), "--port", str(url.port or 80)]
    extra = launch.get("extra_args", [])
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise ValueError("extra_args must be an argv array")
    if any("key" in x.lower() or "token" in x.lower() and "tokens" not in x.lower() for x in extra):
        raise ValueError("Do not store authentication credentials in launch arguments")
    # Prevent silently overriding controlled fields via extra_args.
    reserved = set(mapping.values()) | {x for x in command if x.startswith("--")} | {
        "--config", "--tokenizer", "--tokenizer-path", "--tokenizer-revision", "--model", "--model-path",
        "--revision", "--served-model-name", "--seed", "--random-seed", "--tensor_parallel_size",
        "--tensor-parallel-size", "--free_gpu_memory_fraction", "--trust-remote-code", "--trust_remote_code"}
    if any(x.split("=")[0] in reserved for x in extra):
        raise ValueError("extra_args cannot override an already controlled launch option")
    return command + extra


def verified_metadata(config):
    metadata = read_json(config["server"]["metadata_path"])
    expected = digest(serving_identity(config))
    if metadata.get("serving_fingerprint") != expected or digest(metadata["identity"]) != expected:
        raise ValueError("Serving provenance does not match the model revision, hardware profile, or launch settings")
    if metadata.get("status") != "ready":
        raise ValueError("Serving provenance is not ready; start the configured server")
    return metadata


def configure_serving_environment(config):
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    launch = config.get("launch", {})
    overrides = launch.get("env", {})
    if any(k in ("HOME", "CODEX_HOME") or any(s in k.upper() for s in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")) for k in overrides):
        raise ValueError("Keep credentials and system directories out of launch.env")
    os.environ.update({k: str(v) for k, v in overrides.items()})
    # Ray expects a HIP mask. Keep the scheduler's ROCr restriction and select
    # logical GPU 0 only when it exposes a single device and no mask conflicts.
    if (config.get("hardware", {}).get("vendor") == "amd"
            and config.get("backend") == "vllm"
            and "HIP_VISIBLE_DEVICES" not in os.environ
            and re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", os.environ.get("ROCR_VISIBLE_DEVICES", ""))
            and all(os.environ.get(key) in (None, "0")
                    for key in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"))):
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        event("Set HIP_VISIBLE_DEVICES=0 within the existing single-GPU ROCr allocation",
              stage="startup")
    if launch.get("cuda_version"):
        version = launch["cuda_version"]
        if version not in ("12.8.1", "13.0.2") or config["hardware"]["vendor"] != "nvidia":
            raise ValueError("Workspace CUDA compiler selection requires NVIDIA and a supported toolkit version")
        cuda = ROOT / ".tools" / f"cuda-{version}"
        if not (cuda / "bin/nvcc").exists():
            raise RuntimeError(f"Run scripts/install_cuda_compiler.py --version {version} first")
        os.environ["CUDA_HOME"] = str(cuda)
        os.environ["PATH"] = str(cuda / "bin") + os.pathsep + os.environ.get("PATH", "")
    if launch.get("workspace_gcc"):
        compiler_dir = ROOT / ".tools" / "gcc13"
        gcc = compiler_dir / "bin/x86_64-conda-linux-gnu-gcc"
        gxx = compiler_dir / "bin/x86_64-conda-linux-gnu-g++"
        if not gcc.exists() or not gxx.exists():
            raise RuntimeError("Install the workspace GCC 13 toolchain first; run scripts/install_workspace_gcc.py")
        os.environ.update(CC=str(gcc), CXX=str(gxx), NVCC_CCBIN=str(gxx))
    if launch.get("workspace_gcc"):
        os.environ["LD_LIBRARY_PATH"] = str(ROOT / ".tools/gcc13/lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def serve(config_path, dry_run=False):
    local_environment()
    config = read_json(config_path)
    command = launch_command(config)
    if dry_run:
        print(shlex.join(command))
        return command
    with activity("Checking serving hardware and environment", stage="startup", setting=config.get("setup_id")):
        configure_serving_environment(config)
        environment = discover()
        from .software import check_runtime
        audit = check_runtime(config, environment)
        write_json(Path(config['server']['metadata_path']).parent / 'startup-diagnostics.json',
                   {'environment': environment, 'runtime_contract': audit})
        require_hardware(config["hardware"]["vendor"], environment)
        if audit['status'] != 'match':
            raise RuntimeError(f'Serving runtime differs from its declared baseline: {audit["differences"]}. See runtime-contracts.json and startup-diagnostics.json')
        if config["backend"] not in environment["packages"]:
            raise RuntimeError(f"Install {config['backend']} in this Python environment before serving")
    pinned_snapshot = None
    if config["hardware"]["vendor"] == "tenstorrent":
        from huggingface_hub import snapshot_download
        with activity("Resolving pinned model snapshot", stage="model_load", setting=config.get("setup_id"), model=config["model"]):
            pinned_snapshot = snapshot_download(config["model"], revision=config["revision"])
        # Some loaders apply --revision to weights but reload tokenizer/config
        # from the floating Hub model name. A local snapshot pins all files.
        alias = model_snapshot_alias(config)
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            if alias.resolve() != Path(pinned_snapshot).resolve():
                raise ValueError("Model alias points to a different snapshot")
        elif alias.exists():
            raise ValueError("Model alias already exists and is not a snapshot symlink")
        else:
            alias.symlink_to(Path(pinned_snapshot).resolve(), target_is_directory=True)
        command = launch_command(config, str(alias))
    backend = HTTPBackend(config)
    address = urlsplit(backend.base_url)
    with socket.socket(socket.AF_INET6 if address.hostname == "::1" else socket.AF_INET) as check:
        if check.connect_ex((address.hostname, address.port or 80)) == 0:
            raise RuntimeError("Port is already in use; refusing to attach provenance to someone else's server")
    launch = config.get("launch", {})
    child_env = dict(os.environ)
    identity = serving_identity(config)
    metadata = {"schema_version": 1, "status": "starting", "created_at": now(), "instance_id": str(uuid.uuid4()),
                "identity": identity, "serving_fingerprint": digest(identity), "environment": environment,
                "command": command, "provenance_source": "launcher_on_serving_host"}
    if pinned_snapshot:
        metadata["pinned_snapshot"] = pinned_snapshot
    metadata_path = Path(config["server"]["metadata_path"])
    write_json(metadata_path, metadata)
    log_path = Path(launch.get("log_path", str(metadata_path.with_suffix(".log"))))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_handler = signal.getsignal(signal.SIGTERM)
    def stopped(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)
    process = None
    try:
        with log_path.open("a") as log:
            process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            metadata.update(pid=process.pid, launcher_pid=os.getpid())
            write_json(metadata_path, metadata)
            deadline = time.monotonic() + launch.get("startup_timeout_seconds", 900)
            backend.timeout = 3
            with activity("Waiting for server health check", stage="startup", setting=config.get("setup_id"),
                          framework=config["backend"], log=str(log_path), timeout_seconds=launch.get("startup_timeout_seconds", 900)):
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited with code {process.returncode}; see {log_path}")
                    try:
                        probe = backend.probe()
                        break
                    except (RuntimeError, ValueError, TimeoutError, OSError):
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"Server did not become ready; see {log_path}")
                        time.sleep(2)
            backend.timeout = config["server"].get("timeout_seconds", 900)
            with activity("Validating generation endpoint", stage="startup", setting=config.get("setup_id")):
                validation = backend.generate_one({"input_ids": [1]}, max_tokens=2)
            metadata["effective_context_limit"] = config["engine"]["max_model_len"]
            metadata.update(status="ready", ready_at=now(), probe=probe, pid=process.pid,
                            api_validation={"input_tokens": 1, "output_tokens": validation["output_tokens"],
                                            "token_ids_source": validation["token_ids_source"]})
            write_json(metadata_path, metadata)
            event("Server ready", stage="startup", setting=config.get("setup_id"), url=backend.base_url,
                  context_tokens=metadata["effective_context_limit"], log=str(log_path), provenance=str(metadata_path))
            code = process.wait()
            if code:
                raise RuntimeError(f"Server exited with code {code}; see {log_path}")
    except KeyboardInterrupt:
        metadata["status"] = "stopped"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        event("Stopping owned server process", stage="shutdown", setting=config.get("setup_id"))
        if process is not None:
            # Clean up only the process group created by this launcher.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        if metadata["status"] == "ready":
            metadata["status"] = "stopped"
        metadata["stopped_at"] = now()
        write_json(metadata_path, metadata)
        signal.signal(signal.SIGTERM, old_handler)
        event("Server stopped", stage="shutdown", setting=config.get("setup_id"), status=metadata["status"])
