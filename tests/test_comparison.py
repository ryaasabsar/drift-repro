import copy
import json

import pytest

from driftbench_runner.common import append_jsonl, digest, write_json
from driftbench_runner.comparison import compare_runs


def make_run(directory, answers, labels, gpu="RTX 3060", workload="math"):
    config = {"model": "fixture/test-model", "revision": "abc123", "backend": "vllm", "engine": {"dtype": "bfloat16"},
              "generation": {"temperature": 0}, "prompt_format": "chat", "seed": 42, "batch_size": 1}
    manifest = {"config": config, "environment": {"gpu_name": gpu}, "sources": {workload: {"selected": len(answers)}},
                "run_fingerprint": digest(config), "status": "complete", "expected_records": len(answers)}
    write_json(directory / "manifest.json", manifest)
    evaluations = []
    for i, (answer, label) in enumerate(zip(answers, labels)):
        row = {"workload": workload, "prompt_id": str(i), "source_sha256": digest(i), "request_sha256": digest(i),
               "output_text": answer, "output_token_ids": [ord(c) for c in answer], "finish_reason": "stop"}
        append_jsonl(directory / f"{workload}.jsonl", row)
        evaluations.append({"workload": workload, "prompt_id": str(i), "output_sha256": digest(answer),
                            "correct": label, "score": float(label), "status": "scored", "method": "exact",
                            "evaluator_version": "test"})
    write_json(directory / "evaluation.json", {"summary": {"run_fingerprint": digest(config)}, "records": evaluations})
    return directory


def test_label_flip_differs_from_text_change(tmp_path):
    a = make_run(tmp_path / "a", ["The answer is 5", "6"], [True, False])
    b = make_run(tmp_path / "b", ["5", "5"], [True, True], gpu="A100")
    report = compare_runs(a, b, tmp_path / "comparison")
    summary = report["workloads"]["math"]
    assert summary["text_change_rate"] == 1
    assert summary["flip_rate"] == 0.5
    assert summary["incorrect_to_correct"] == 1
    assert report["changed_factors"] == ["hardware"]


def test_missing_prompt_not_silently_dropped(tmp_path):
    a = make_run(tmp_path / "a", ["1", "2"], [True, False])
    b = make_run(tmp_path / "b", ["1"], [True])
    with pytest.raises(ValueError, match="identical prompt-ID"):
        compare_runs(a, b, tmp_path / "comparison")


def test_changed_decoding_refused(tmp_path):
    a = make_run(tmp_path / "a", ["1"], [True])
    b = make_run(tmp_path / "b", ["1"], [True])
    manifest = json.loads((b / "manifest.json").read_text())
    manifest["config"]["generation"]["temperature"] = 1.0
    write_json(b / "manifest.json", manifest)
    with pytest.raises(ValueError, match="experimental controls"):
        compare_runs(a, b, tmp_path / "comparison")


def test_stale_evaluation_rejected(tmp_path):
    a = make_run(tmp_path / "a", ["1"], [True])
    b = make_run(tmp_path / "b", ["2"], [False])
    evaluation = json.loads((b / "evaluation.json").read_text())
    evaluation["records"][0]["output_sha256"] = digest("old output")
    write_json(b / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Stale evaluation"):
        compare_runs(a, b, tmp_path / "comparison")


def test_judge_hardware_must_match_even_with_confounds_allowed(tmp_path):
    a = make_run(tmp_path / "a", ["I cannot assist."], [True], workload="safety")
    b = make_run(tmp_path / "b", ["I cannot assist."], [True], workload="safety")
    for directory, gpu in ((a, "RTX 3060"), (b, "A100")):
        evaluation = json.loads((directory / "evaluation.json").read_text())
        evaluation["records"][0].update(method="safety_classification", label="safe",
                                         judge={"model": "same-judge", "revision": "same-revision",
                                                "environment": {"cuda_devices": [gpu]}})
        write_json(directory / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Evaluators differ.*judge"):
        compare_runs(a, b, tmp_path / "comparison", allow_confounded=True)


def test_identical_output_cannot_be_reported_as_a_correctness_flip(tmp_path):
    a = make_run(tmp_path / "a", ["5"], [True])
    b = make_run(tmp_path / "b", ["5"], [False])
    with pytest.raises(ValueError, match="Identical response received different labels"):
        compare_runs(a, b, tmp_path / "comparison")


def test_code_execution_hosts_must_match(tmp_path):
    a = make_run(tmp_path / "a", ["def f(): return 1"], [True], workload="code")
    b = make_run(tmp_path / "b", ["def f(): return 2"], [False], workload="code")
    for directory, cpu in ((a, "CPU A"), (b, "CPU B")):
        evaluation = json.loads((directory / "evaluation.json").read_text())
        evaluation["records"][0].update(method="humaneval_execution_pass_at_1", execution_environment={"cpu": cpu})
        write_json(directory / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Evaluators differ.*execution_environment"):
        compare_runs(a, b, tmp_path / "comparison", allow_confounded=True)
