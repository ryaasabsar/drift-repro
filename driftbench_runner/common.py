import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "c915e781a17c2d4c9bd768e60a1bc9734c5f2895"
DATASET_DIR = ROOT / "vendor/driftbench-ae/artifact_evaluation/reviewer-verification-master/datasets"
WORKLOADS = {
    "code": ("humaneval_prompts.jsonl", 164),
    "math": ("gsm8k_prompts.jsonl", 500),
    "safety": ("advbench_prompts.jsonl", 520),
    "chat": ("lmsys_prompts.jsonl", 1000),
    "long_context": ("long_qa_prompts.jsonl", 100),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    # JSONL uses physical newlines. str.splitlines() also splits valid Unicode
    # separators inside JSON strings, corrupting multilingual prompts/outputs.
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def keyed(rows):
    result = {}
    for row in rows:
        key = (row["workload"], row["prompt_id"])
        if key in result:
            raise ValueError(f"Duplicate prompt: {key}")
        result[key] = row
    return result


def load_workloads(names, limit=None, dataset_dir=DATASET_DIR):
    rows, sources = [], {}
    for workload in names:
        filename, expected = WORKLOADS[workload]
        path = Path(dataset_dir) / filename
        data = read_jsonl(path)
        if len(data) != expected:
            raise ValueError(f"{path}: expected {expected} records, got {len(data)}")
        sources[workload] = {"file": filename, "sha256": digest(path.read_bytes()), "available": len(data)}
        selected = data if limit is None else data[:limit]
        sources[workload]["selected"] = len(selected)
        for row in selected:
            rows.append({**row, "workload": workload, "source_sha256": digest(row)})
    keyed(rows)
    return rows, sources


def local_environment():
    # All mutable caches stay inside the workspace.
    for key, directory in {
        "HF_HOME": ".cache/huggingface", "XDG_CACHE_HOME": ".cache",
        "VLLM_CACHE_ROOT": ".cache/vllm", "TRITON_CACHE_DIR": ".cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
        "TVM_FFI_CACHE_DIR": ".cache/tvm-ffi", "FLASHINFER_WORKSPACE_BASE": ".cache/flashinfer",
        "TORCH_EXTENSIONS_DIR": ".cache/torch-extensions",
    }.items():
        os.environ.setdefault(key, str(ROOT / directory))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
