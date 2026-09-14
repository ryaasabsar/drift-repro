import importlib.metadata
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from .logging import InferenceProgress, activity, event

from .common import (append_jsonl, digest, keyed, load_workloads, local_environment,
                     now, read_json, read_jsonl, write_json, UPSTREAM_COMMIT)


def environment_metadata():
    packages = {}
    for name in ("vllm", "torch", "transformers", "tokenizers", "triton", "sentence-transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    from .hardware import command_output
    gpu = command_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
                          "--format=csv,noheader"])
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "nvidia_smi": gpu.get("stdout", ""), "nvidia_smi_error": gpu.get("stderr", "")}


def render_prompt(tokenizer, row, config):
    if config["prompt_format"] == "raw":
        return row["prompt"]
    if config["prompt_format"] != "chat":
        raise ValueError("prompt_format must be chat or raw")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=False,
        add_generation_prompt=True, **config.get("chat_template_kwargs", {}))


def planned_batches(rows, config):
    # Preserve original batch membership on resume, including partially saved
    # batches. Batch shape is an experimental control in a drift study.
    for workload in dict.fromkeys(r["workload"] for r in rows):
        workload_rows = [r for r in rows if r["workload"] == workload]
        size = config.get("batch_size_by_workload", {}).get(workload, config["batch_size"])
        if not isinstance(size, int) or size < 1:
            raise ValueError("Every batch size must be a positive integer")
        for start in range(0, len(workload_rows), size):
            yield workload_rows[start:start + size]


def prepare(config_path, workload_names, limit=None):
    local_environment()
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    config = read_json(config_path)
    if config["batch_size"] < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size and limit must be positive")
    # Resolve floating revisions once, then store and use the immutable revision.
    if Path(config["model"]).is_dir():
        raise ValueError("Use a Hub model ID with a revision for verifiable weight provenance")
    revision = config["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().model_info(config["model"], revision=revision).sha
    config["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=revision)
    rows, sources = load_workloads(workload_names, limit)
    for row in rows:
        row["rendered_prompt"] = render_prompt(tokenizer, row, config)
        row["input_ids"] = tokenizer.encode(row["rendered_prompt"], add_special_tokens=False)
        row["request_sha256"] = digest({"input_ids": row["input_ids"], "prompt": row["rendered_prompt"]})
    return config, tokenizer, rows, sources


def run(config_path, output_dir, workload_names, limit=None, resume=False, preflight=False):
    with activity("Loading tokenizer and preparing prompts", stage="preparation", config=str(config_path)):
        config, tokenizer, rows, sources = prepare(config_path, workload_names, limit)
    from .hardware import validate_target
    validate_target(config)
    max_input = max(len(row["input_ids"]) for row in rows)
    max_total = max_input + config["generation"]["max_tokens"]
    event("Prompts prepared", stage="preparation", setting=config.get("setup_id"), total=len(rows),
          longest_input_tokens=max_input, required_context=max_total)
    if max_total > config["engine"]["max_model_len"]:
        raise ValueError(f"Need max_model_len >= {max_total}; no inputs will be silently truncated")
    if preflight:
        return {"config": config, "sources": sources, "max_input_tokens": max_input,
                "required_context": max_total}
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    # A file lock prevents concurrent writers, including during resume.
    import fcntl
    with (outdir / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if config.get("transport", "offline") == "http":
            from .http_inference import run_http
            return run_http(config, rows, sources, outdir, resume)
        return _run_locked(config, rows, sources, outdir, resume)


def _run_locked(config, rows, sources, outdir, resume):
    manifest_path = outdir / "manifest.json"
    identity = {"config": config, "sources": sources,
                "requests": [[r["workload"], r["prompt_id"], r["request_sha256"], r["source_sha256"]] for r in rows]}
    run_fingerprint = digest(identity)
    if manifest_path.exists():
        old = read_json(manifest_path)
        if not resume:
            raise ValueError("Run exists; choose another directory or use --resume")
        if old["run_fingerprint"] != run_fingerprint:
            raise ValueError("Resume refused: config, model revision, dataset or prompts changed")
        current_environment = environment_metadata()
        for field in ("python", "packages", "nvidia_smi"):
            if old["environment"].get(field) != current_environment.get(field):
                raise ValueError(f"Resume refused: environment changed ({field})")
        manifest = old
    else:
        if any(outdir.glob("*.jsonl")):
            raise ValueError("Output records exist without a manifest")
        manifest = {"schema_version": 1, "created_at": now(), "upstream_commit": UPSTREAM_COMMIT,
                    "run_fingerprint": run_fingerprint, "config": config, "sources": sources,
                    "expected_records": len(rows), "environment": environment_metadata(),
                    "status": "prepared"}
    existing = keyed([r for name in sources for r in read_jsonl(outdir / f"{name}.jsonl")])
    expected = keyed(rows)
    for key, record in existing.items():
        if key not in expected or record["request_sha256"] != expected[key]["request_sha256"]:
            raise ValueError(f"Invalid existing output record: {key}")
    pending = [r for r in rows if (r["workload"], r["prompt_id"]) not in existing]
    if not pending:
        manifest.update(status="complete", completed_records=len(rows), updated_at=now())
        manifest.pop("error", None)
        write_json(manifest_path, manifest)
        event("Reusing all saved responses", stage="resume", setting=config.get("setup_id"), completed=len(rows), total=len(rows))
        return manifest
    if config["backend"] != "vllm":
        raise ValueError("This runner currently implements the vLLM backend")
    manifest.pop("error", None)
    manifest.update(status="loading", updated_at=now())
    manifest.setdefault("attempts", []).append({"started_at": now(), "already_completed": len(existing), "remaining": len(pending)})
    write_json(manifest_path, manifest)
    try:
        with activity("Loading inference model", stage="model_load", setting=config.get("setup_id"), model=config["model"]):
            import torch
            from vllm import LLM, SamplingParams
            if config.get("hardware", {}).get("vendor") == "amd":
                from .hardware import discover, require_hardware
                hardware = discover()
                require_hardware("amd", hardware)
                if manifest["environment"].get("rocm") not in (None, hardware):
                    raise ValueError("Resume refused: ROCm serving environment changed")
                manifest["environment"]["rocm"] = hardware
                manifest["environment"]["rocm_runtime"] = torch.version.hip
            manifest["environment"].update(cuda_runtime=torch.version.cuda,
                                           gpu_name=torch.cuda.get_device_name(0),
                                           gpu_capability=list(torch.cuda.get_device_capability(0)))
            llm = LLM(model=config["model"], revision=config["revision"], tokenizer_revision=config["revision"],
                      seed=config["seed"], generation_config="vllm", **config["engine"])
        params = SamplingParams(seed=config["seed"], **config["generation"])
        # Warmup is outside all measured records.
        with activity("Warming up model", stage="warmup", setting=config.get("setup_id")):
            llm.generate([{"prompt_token_ids": rows[0]["input_ids"][:128]}],
                         SamplingParams(temperature=0, max_tokens=2, ignore_eos=True), use_tqdm=False)
        manifest.update(status="running", updated_at=now())
        write_json(manifest_path, manifest)
        completed = len(existing)
        with InferenceProgress(rows, existing, config) as progress:
            for batch_index, batch in enumerate(planned_batches(rows, config)):
                if all((r["workload"], r["prompt_id"]) in existing for r in batch):
                    continue
                progress.batch(batch, batch_index)
                begin = time.perf_counter()
                outputs = llm.generate([{"prompt_token_ids": r["input_ids"]} for r in batch], params, use_tqdm=False)
                elapsed = time.perf_counter() - begin
                if len(outputs) != len(batch):
                    raise RuntimeError("Output count does not match input count")
                for row, output in zip(batch, outputs):
                    if (row["workload"], row["prompt_id"]) in existing:
                        continue
                    completion = output.outputs[0]
                    record = {"schema_version": 1, "setup_id": config["setup_id"], "workload": row["workload"],
                              "prompt_id": row["prompt_id"], "source_sha256": row["source_sha256"],
                              "request_sha256": row["request_sha256"], "prompt": row["prompt"],
                              "rendered_prompt": row["rendered_prompt"], "input_token_ids": row["input_ids"],
                              "output_text": completion.text, "output_token_ids": list(completion.token_ids),
                              "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
                              "input_tokens": len(row["input_ids"]), "output_tokens": len(completion.token_ids),
                              "batch_wall_seconds": elapsed, "batch_size": len(batch), "batch_index": batch_index,
                              "latency_seconds": elapsed if len(batch) == 1 else None,
                              "status": "ok", "timestamp": now()}
                    append_jsonl(outdir / f"{row['workload']}.jsonl", record)
                    completed += 1
                progress.saved(batch, elapsed)
        manifest.update(status="complete", completed_at=now(), completed_records=len(rows))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", updated_at=now())
        raise
    finally:
        write_json(manifest_path, manifest)
    return manifest
