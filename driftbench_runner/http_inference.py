"""Inference on a local or remote server, preserving the existing evaluation schema."""
from pathlib import Path
import time

from .common import UPSTREAM_COMMIT, append_jsonl, digest, keyed, now, read_json, read_jsonl, write_json
from .http_backend import HTTPBackend
from .inference import planned_batches
from .serving import verified_metadata
from .logging import InferenceProgress, activity, event
from .software import client_environment


def run_http(config, rows, sources, outdir, resume=False):
    with activity("Checking serving endpoint and provenance", stage="connection", setting=config["setup_id"]):
        metadata = verified_metadata(config)
        required = max(len(r["input_ids"]) for r in rows) + config["generation"]["max_tokens"]
        if required > metadata.get("effective_context_limit", config["engine"]["max_model_len"]):
            raise ValueError(f"Server's effective context limit is below the required {required} tokens")
        backend = HTTPBackend(config)
        probe = backend.probe()
    # The sidecar is a serving-host observation. /v1/models verifies the alias,
    # not the weights: do not describe the endpoint as remote attestation.
    identity = {"config": config, "sources": sources, "server": metadata["serving_fingerprint"],
                "requests": [[r["workload"], r["prompt_id"], r["request_sha256"], r["source_sha256"]] for r in rows]}
    fingerprint = digest(identity)
    path = outdir / "manifest.json"
    environment = dict(metadata["environment"])
    client = client_environment()
    environment["gpu_name"] = ", ".join(x["name"] for x in environment.get("accelerators", [])) or config["hardware"]["device"]
    if path.exists():
        manifest = read_json(path)
        if not resume:
            raise ValueError("Run exists; choose another directory or use --resume")
        if manifest["run_fingerprint"] != fingerprint or manifest["environment"] != environment:
            raise ValueError("Resume refused: requests, configuration, or serving environment changed")
        if manifest.get('client_environment') != client:
            raise ValueError('Resume refused: tokenizer client or runner source changed; use a new run ID')
    else:
        if any(outdir.glob("*.jsonl")):
            raise ValueError("Output records exist without a manifest")
        manifest = {"schema_version": 2, "created_at": now(), "upstream_commit": UPSTREAM_COMMIT,
                    "config": config, "sources": sources, "run_fingerprint": fingerprint,
                    "expected_records": len(rows), "environment": environment,
                    "server_provenance": metadata, "scheduling": "client_concurrent_barrier",
                    "client_environment": client,
                    "status": "prepared"}
    existing = keyed([r for name in sources for r in read_jsonl(outdir / f"{name}.jsonl")])
    expected = keyed(rows)
    for key, record in existing.items():
        if key not in expected or any(record[field] != expected[key][field] for field in ("request_sha256", "source_sha256")):
            raise ValueError(f"Invalid existing output record: {key}")
    if len(existing) == len(rows):
        manifest.update(status="complete", completed_records=len(rows), updated_at=now())
        manifest.pop("error", None)
        write_json(path, manifest)
        event("Reusing all saved responses", stage="resume", setting=config["setup_id"], completed=len(rows), total=len(rows))
        return manifest
    # A new server instance may change cache state; record it rather than conceal it.
    manifest.setdefault("attempts", []).append({"started_at": now(), "already_completed": len(existing),
                                               "server_instance": metadata["instance_id"], "probe": probe})
    manifest.pop("error", None)
    manifest.update(status="running", updated_at=now())
    write_json(path, manifest)
    try:
        with activity("Warming up serving endpoint", stage="warmup", setting=config["setup_id"]):
            backend.generate_one({"input_ids": rows[0]["input_ids"][:128]}, max_tokens=2)
        completed = len(existing)
        with InferenceProgress(rows, existing, config) as progress:
            for index, batch in enumerate(planned_batches(rows, config)):
                if all((r["workload"], r["prompt_id"]) in existing for r in batch):
                    continue
                progress.batch(batch, index)
                begin = time.perf_counter()
                outputs = backend.generate(batch)
                elapsed = time.perf_counter() - begin
                for row, output in zip(batch, outputs):
                    if (row["workload"], row["prompt_id"]) in existing:
                        continue
                    record = {"schema_version": 2, "setup_id": config["setup_id"],
                              **{k: row[k] for k in ("workload", "prompt_id", "source_sha256", "request_sha256", "prompt", "rendered_prompt")},
                              "input_token_ids": row["input_ids"], "input_tokens": len(row["input_ids"]), **output,
                              "batch_wall_seconds": elapsed, "batch_size": len(batch), "batch_index": index,
                              "scheduling": "client_concurrent_barrier", "server_instance": metadata["instance_id"],
                              "status": "ok", "timestamp": now()}
                    append_jsonl(outdir / f"{row['workload']}.jsonl", record)
                    completed += 1
                progress.saved(batch, elapsed)
        manifest.update(status="complete", completed_records=completed, completed_at=now())
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", updated_at=now())
        raise
    finally:
        write_json(path, manifest)
    return manifest
