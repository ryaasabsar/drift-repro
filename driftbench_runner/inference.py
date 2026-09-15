import re
from pathlib import Path
from .logging import activity, event

from .common import digest, load_workloads, local_environment, read_json


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
        raise ValueError("Use an HTTP serving config; legacy offline experiments are archived")
