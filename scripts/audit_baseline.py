"""Integrity audit for the completed RTX 3060 baseline; run from the project."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftbench_runner.common import (
    ROOT, digest, keyed, load_workloads, now, read_json, read_jsonl, write_json,
)
from driftbench_runner.inference import planned_batches
from driftbench_runner.evaluation import EVALUATOR_VERSION

directory = ROOT / "results/rtx3060-qwen35-vllm"
manifest = read_json(directory / "manifest.json")
assert manifest["status"] == "complete", "Inference is incomplete"
assert manifest["completed_records"] == manifest["expected_records"] == 2284
source_rows, sources = load_workloads(list(manifest["sources"]))
assert sources == manifest["sources"]
reference = keyed(source_rows)
outputs = keyed([r for w in sources for r in read_jsonl(directory / f"{w}.jsonl")])
assert set(outputs) == set(reference), "Missing, duplicate or unexpected input IDs"
config = manifest["config"]
assert config == read_json(ROOT / "configs/rtx3060-qwen35-vllm-batch16.json")
batches = {}
for index, batch in enumerate(planned_batches(source_rows, config)):
    for row in batch:
        batches[(row["workload"], row["prompt_id"])] = (index, len(batch))
for key, row in outputs.items():
    assert row["status"] == "ok"
    assert row["setup_id"] == config["setup_id"]
    assert row["prompt"] == reference[key]["prompt"]
    assert row["source_sha256"] == reference[key]["source_sha256"]
    assert row["request_sha256"] == digest({"input_ids": row["input_token_ids"], "prompt": row["rendered_prompt"]})
    assert row["input_tokens"] == len(row["input_token_ids"])
    assert row["output_tokens"] == len(row["output_token_ids"]) <= config["generation"]["max_tokens"]
    assert row["input_tokens"] + config["generation"]["max_tokens"] <= config["engine"]["max_model_len"]
    assert (row["batch_index"], row["batch_size"]) == batches[key]
    assert row["finish_reason"] in ("stop", "length")
    assert math.isfinite(row["batch_wall_seconds"]) and row["batch_wall_seconds"] > 0
identity = {"config": config, "sources": sources, "requests": [
    [r["workload"], r["prompt_id"], outputs[(r["workload"], r["prompt_id"])]["request_sha256"], r["source_sha256"]]
    for r in source_rows
]}
assert digest(identity) == manifest["run_fingerprint"]

evaluation = read_json(directory / "evaluation.json")
assert evaluation["summary"]["run_fingerprint"] == manifest["run_fingerprint"]
assert evaluation["summary"]["evaluator_version"] == EVALUATOR_VERSION
scores = keyed(evaluation["records"])
assert set(scores) == set(outputs)
for key, score in scores.items():
    assert score["output_sha256"] == digest(outputs[key]["output_text"])
    assert score["source_sha256"] == outputs[key]["source_sha256"]
    assert score["status"] == ("pairwise_only" if key[0] == "chat" else "scored")
for workload, summary in evaluation["summary"]["workloads"].items():
    rows = [r for (w, _), r in outputs.items() if w == workload]
    assert summary["generated"] == summary["expected"] == sources[workload]["selected"] == len(rows)
    assert summary["missing"] == summary["pending"] == 0
    assert summary["length_limited"] == sum(r["finish_reason"] == "length" for r in rows)
    for field in ("input_tokens", "output_tokens"):
        assert summary[field] == sum(r[field] for r in rows)

labels = keyed(read_jsonl(directory / "safety-labels.jsonl"))
assert set(labels) == {key for key in outputs if key[0] == "safety"}
judge_hashes = set()
for key, label in labels.items():
    assert label["output_sha256"] == digest(outputs[key]["output_text"])
    assert label["source_sha256"] == outputs[key]["source_sha256"]
    assert label["label"] == scores[key]["label"]
    assert label["label"] == ("safe" if label["severity"] == "safe" else "unsafe")
    assert label["severity"] in ("safe", "unsafe", "controversial")
    assert label["judge"] == scores[key]["judge"]
    assert label["judge"]["model"] == "Qwen/Qwen3Guard-Gen-0.6B"
    assert label["judge"]["revision"] == read_json(ROOT / "models.lock.json")["safety"]["revision"]
    judge_hashes.add(digest(label["judge"]))
assert len(judge_hashes) == 1
assert len({digest(r["execution_environment"]) for r in scores.values() if r["workload"] == "code"}) == 1

report = {
    "audited_at": now(), "status": "passed", "run_fingerprint": manifest["run_fingerprint"],
    "inference_records": len(outputs), "objective_scores": sum(r["status"] == "scored" for r in scores.values()),
    "chat_records_for_future_pairing": sum(r["status"] == "pairwise_only" for r in scores.values()),
    "safety_judgments": len(labels), "fixed_batches": len(set(batches.values())),
    "checks": ["exact dataset ID sets and source hashes", "original prompts and request hashes",
               "token counts and context limits", "fixed batch membership", "run/config fingerprint",
               "complete evaluation coverage and matching response hashes", "common safety-judge provenance",
               "common code-test environment", "summary counts and token totals"],
    "sha256": {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir())
               if p.suffix in (".json", ".jsonl", ".csv") and p.name != "audit.json"},
}
assert report["objective_scores"] == 1284 and report["chat_records_for_future_pairing"] == 1000
write_json(directory / "audit.json", report)
print({k: v for k, v in report.items() if k not in ("sha256", "checks")})
