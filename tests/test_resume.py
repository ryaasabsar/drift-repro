import pytest

from driftbench_runner.common import append_jsonl, digest, read_json, read_jsonl, write_json
from driftbench_runner.inference import _run_locked, planned_batches


def test_jsonl_roundtrip_preserves_unicode_line_separators(tmp_path):
    path = tmp_path / "responses.jsonl"
    rows = [{"prompt": "first\u2028second\u2029third", "output": "a\u0085b\nnew line"},
            {"prompt": "another prompt", "output": "another response"}]
    for row in rows:
        append_jsonl(path, row)
    assert read_jsonl(path) == rows


def prepared_run(tmp_path, monkeypatch):
    config = {"model": "test", "backend": "vllm"}
    rows = [{"workload": "math", "prompt_id": "1", "request_sha256": "request", "source_sha256": "source"}]
    sources = {"math": {"selected": 1}}
    identity = {"config": config, "sources": sources, "requests": [["math", "1", "request", "source"]]}
    environment = {"python": "3.12", "packages": {}, "nvidia_smi": "test GPU"}
    monkeypatch.setattr("driftbench_runner.inference.environment_metadata", lambda: environment)
    write_json(tmp_path / "manifest.json", {"run_fingerprint": digest(identity), "environment": dict(environment), "status": "interrupted"})
    append_jsonl(tmp_path / "math.jsonl", rows[0])
    return config, rows, sources, environment


def test_resume_recovers_after_final_record_before_manifest_update(tmp_path, monkeypatch):
    config, rows, sources, _ = prepared_run(tmp_path, monkeypatch)
    _run_locked(config, rows, sources, tmp_path, True)
    assert read_json(tmp_path / "manifest.json")["status"] == "complete"


def test_resume_refuses_another_gpu(tmp_path, monkeypatch):
    config, rows, sources, environment = prepared_run(tmp_path, monkeypatch)
    environment["nvidia_smi"] = "another GPU"
    with pytest.raises(ValueError, match="environment changed"):
        _run_locked(config, rows, sources, tmp_path, True)


def test_resume_refuses_changed_generation_settings(tmp_path, monkeypatch):
    config, rows, sources, _ = prepared_run(tmp_path, monkeypatch)
    config["generation"] = {"temperature": 1}
    with pytest.raises(ValueError, match="config, model revision"):
        _run_locked(config, rows, sources, tmp_path, True)


def test_batch_plan_respects_workload_boundaries_and_context_override():
    rows = [{"workload": "math", "prompt_id": str(i)} for i in range(5)]
    rows += [{"workload": "long_context", "prompt_id": str(i)} for i in range(2)]
    config = {"batch_size": 4, "batch_size_by_workload": {"long_context": 1}}
    batches = list(planned_batches(rows, config))
    assert [len(b) for b in batches] == [4, 1, 1, 1]
    assert [[r["prompt_id"] for r in b] for b in batches] == [["0", "1", "2", "3"], ["4"], ["0"], ["1"]]
