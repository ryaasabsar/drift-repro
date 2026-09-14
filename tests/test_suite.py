import json
from pathlib import Path

import pytest

from driftbench_runner import suite
from driftbench_runner.common import ROOT, digest, load_workloads, now, read_json, write_json
from driftbench_runner.evaluation import EVALUATOR_VERSION


@pytest.fixture(autouse=True)
def isolated_suite_locks(tmp_path, monkeypatch):
    # Unit tests must not acquire the real experiment's accelerator lock.
    monkeypatch.setattr(suite, "ROOT", tmp_path)


def specification(tmp_path):
    config = read_json(ROOT / "configs/rtx3060-qwen35-vllm.json")
    write_json(tmp_path / "profile.json", config)
    path = tmp_path / "suite.json"
    write_json(path, {"suite_id": "test", "workloads": ["math"], "limit": 1, "settings": [
        {"id": sid, "config": "profile.json", "server_python": str(ROOT / ".venv/bin/python")}
        for sid in ("gpu_model_vllm", "gpu_model_sglang")]})
    return path


def save_inference(item, plan):
    directory = Path(item["run_dir"])
    inputs, sources = load_workloads(plan["workloads"], plan["limit"])
    records = [{"workload": row["workload"], "prompt_id": row["prompt_id"], "source_sha256": row["source_sha256"],
                "prompt": row["prompt"], "rendered_prompt": row["prompt"], "input_token_ids": [1],
                "request_sha256": digest({"input_ids": [1], "prompt": row["prompt"]}),
                "status": "ok", "output_text": "18", "output_token_ids": [2], "finish_reason": "stop"}
               for row in inputs]
    identity = {"config": item["config"], "sources": sources,
                "requests": [[r["workload"], r["prompt_id"], r["request_sha256"], r["source_sha256"]] for r in records]}
    write_json(directory / "manifest.json", {"config": item["config"], "sources": sources, "environment": {"gpu_name": "test"},
               "status": "complete", "expected_records": len(records), "run_fingerprint": digest(identity)})
    (directory / "math.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def save_scores(directory):
    manifest = read_json(directory / "manifest.json")
    rows = [json.loads(line) for line in (directory / "math.jsonl").read_text().splitlines()]
    scores = [{"workload": r["workload"], "prompt_id": r["prompt_id"], "output_sha256": digest(r["output_text"]),
               "source_sha256": r["source_sha256"], "status": "scored", "correct": True, "score": 1.0,
               "method": "numeric_exact_match", "evaluator_version": EVALUATOR_VERSION} for r in rows]
    write_json(directory / "evaluation.json", {"summary": {"run_fingerprint": manifest["run_fingerprint"]}, "records": scores})


def test_full_preset_and_subset_plan_are_explicit(tmp_path):
    full = suite.load_plan(ROOT / "suites/rtx3060.json", tmp_path)
    assert full["expected_records_per_setting"] == 2284
    assert len(full["settings"]) == 4
    small = suite.load_plan(ROOT / "suites/rtx3060.json", tmp_path, limit=2, device="0")
    assert small["expected_records_per_setting"] == 10
    assert all(s["config"]["launch"]["env"]["CUDA_VISIBLE_DEVICES"] == "0" for s in small["settings"])
    assert all(s["config"]["server"]["metadata_path"].startswith(str(tmp_path)) for s in small["settings"])
    with pytest.raises(ValueError, match="Unknown settings"):
        suite.run_suite(ROOT / "suites/rtx3060.json", tmp_path, selected=["typo"], dry_run=True)


def test_select_resume_keeps_completed_inference_and_finishes_remaining(tmp_path, monkeypatch):
    path, out, events = specification(tmp_path), tmp_path / "out", []
    def run(item, plan, resume, environment):
        events.append((item["id"], "inference_and_shutdown"))
        save_inference(item, plan)
    def score(command, log, environment):
        directory = Path(command[4])
        events.append((directory.name, "score"))
        save_scores(directory)
    monkeypatch.setattr(suite, "run_inference", run)
    monkeypatch.setattr(suite, "run_command", score)
    first = suite.run_suite(path, out, selected=["gpu_model_vllm"])
    assert first["status"] == "partial"
    old = (out / "settings/gpu_model_vllm/math.jsonl").read_bytes()
    second = suite.run_suite(path, out, resume=True)
    assert second["status"] == "complete"
    assert events == [("gpu_model_vllm", "inference_and_shutdown"), ("gpu_model_vllm", "score"),
                      ("gpu_model_sglang", "inference_and_shutdown"), ("gpu_model_sglang", "score")]
    suite.run_suite(path, out, resume=True)
    assert len(events) == 4
    assert (out / "settings/gpu_model_vllm/math.jsonl").read_bytes() == old
    assert (out / "collection/workloads/math.csv").exists()
    with pytest.raises(ValueError, match="controls changed"):
        suite.run_suite(path, out, limit=2, resume=True)


def test_failed_setting_does_not_hide_completed_setting(tmp_path, monkeypatch):
    path, out = specification(tmp_path), tmp_path / "out"
    def run(item, plan, resume, environment):
        if item["id"] == "gpu_model_vllm":
            raise RuntimeError("GPU unavailable")
        save_inference(item, plan)
    monkeypatch.setattr(suite, "run_inference", run)
    monkeypatch.setattr(suite, "run_command", lambda command, *_: save_scores(Path(command[4])))
    state = suite.run_suite(path, out, continue_on_error=True)
    assert state["status"] == "failed"
    assert state["settings"]["gpu_model_sglang"]["status"] == "complete"
    assert "GPU unavailable" in state["settings"]["gpu_model_vllm"]["error"]
    assert read_json(out / "collection/settings.json")["settings"][0]["setting_id"] == "gpu_model_sglang"


def test_resume_rejects_tampered_saved_request(tmp_path, monkeypatch):
    path, out = specification(tmp_path), tmp_path / "out"
    monkeypatch.setattr(suite, "run_inference", lambda item, plan, *_: save_inference(item, plan))
    suite.run_suite(path, out, inference_only=True)
    file = out / "settings/gpu_model_vllm/math.jsonl"
    row = json.loads(file.read_text())
    row["input_token_ids"] = [999]
    file.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="request trace changed"):
        suite.run_suite(path, out, resume=True)


def test_http_failure_still_stops_own_launcher(tmp_path, monkeypatch):
    plan = suite.load_plan(ROOT / "suites/rtx3060.json", tmp_path, limit=1)
    item = plan["settings"][0]
    Path(item["run_dir"]).mkdir(parents=True)
    stopped = []
    class FakeLauncher:
        pid = 999999
        def __init__(self, *args, **kwargs):
            write_json(item["config"]["server"]["metadata_path"],
                       {"created_at": now(), "launcher_pid": self.pid, "status": "ready"})
        def poll(self):
            return None
    def stop(process, launcher=False):
        stopped.append((process.pid, launcher))
        write_json(item["config"]["server"]["metadata_path"],
                   {"created_at": now(), "launcher_pid": process.pid, "status": "stopped", "stopped_at": now()})
    def fail(*args):
        raise RuntimeError("inference failed")
    monkeypatch.setattr(suite.subprocess, "Popen", FakeLauncher)
    monkeypatch.setattr(suite, "run_command", fail)
    monkeypatch.setattr(suite, "stop_process", stop)
    with pytest.raises(RuntimeError, match="inference failed"):
        suite.run_inference(item, plan, False, {})
    assert stopped == [(999999, True)]


def test_comparison_subprocess_is_logged_and_produces_paired_rows(tmp_path, monkeypatch):
    path, out = specification(tmp_path), tmp_path / 'out'
    spec = read_json(path)
    spec['comparisons'] = [{'baseline': 'gpu_model_vllm', 'candidate': 'gpu_model_sglang'}]
    write_json(path, spec)
    monkeypatch.setattr(suite, 'run_inference', lambda item, plan, *_: save_inference(item, plan))
    actual_command = suite.run_command
    def command(argv, log_path, environment):
        if argv[3] == 'score':
            save_scores(Path(argv[4]))
        else:
            environment = {**environment, 'PYTHONPATH': str(ROOT)}
            actual_command(argv, log_path, environment)
    monkeypatch.setattr(suite, 'run_command', command)
    state = suite.run_suite(path, out)
    assert state['status'] == 'complete'
    directory = out / 'comparisons/gpu_model_vllm__vs__gpu_model_sglang'
    assert (directory / 'rows.html').is_file()
    events = [json.loads(line) for line in (directory / 'logs/comparison.events.jsonl').read_text().splitlines()]
    assert any(e['message'] == 'Comparison saved' and e['paired'] == 1 for e in events)
