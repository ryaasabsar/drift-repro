import csv
import json
import re

import pytest

from driftbench_runner.common import digest, read_json, read_jsonl, write_json
from driftbench_runner.comparison import compare_runs
from driftbench_runner.tables import collect_runs, export_run, outcome
from test_comparison import make_run


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_gather_joins_ids_not_file_positions_and_marks_missing(tmp_path):
    a = make_run(tmp_path / "a", ["correct A", "wrong A"], [True, False])
    b = make_run(tmp_path / "b", ["wrong B"], [False])
    raw = read_jsonl(a / "math.jsonl")
    (a / "math.jsonl").write_text("\n".join(json.dumps(r) for r in reversed(raw)) + "\n")
    rows = export_run(a)
    assert [r["result"] for r in rows] == ["correct", "incorrect"]
    assert [r["row_number"] for r in rows] == [1, 2]
    result = collect_runs([a, b], tmp_path / "gather")
    assert result["rows"] == 3 and result["prompt_rows"] == 2
    matrix = csv_rows(tmp_path / "gather/workloads/math.csv")
    assert matrix[0]["a.output_text"] == "correct A"
    assert matrix[0]["b.output_text"] == "wrong B"
    assert matrix[1]["b.result"] == "missing"
    assert matrix[1]["b.correct"] == ""


def test_csv_and_browser_preserve_full_responses_safely(tmp_path):
    text = 'first, "quoted"\nsecond\u2028third </script><img src=x onerror=alert(1)>'
    a = make_run(tmp_path / "a", [text], [True])
    b = make_run(tmp_path / "b", ["different"], [False])
    out = tmp_path / "pair"
    compare_runs(a, b, out)
    row = csv_rows(out / "rows.csv")[0]
    assert row["baseline_output_text"] == text
    assert row["candidate_result"] == "incorrect" and row["label_flip"] == "True"
    page = (out / "rows.html").read_text()
    assert "<img src=x" not in page
    payload = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)[1]
    assert json.loads(payload)[0]["baseline_output_text"] == text


def test_statuses_are_not_invented_correctness_labels():
    assert outcome({"status": "scored", "label": "unsafe", "correct": False}) == "unsafe"
    for status in ("pending", "error", "pairwise_only"):
        assert outcome({"status": status}) == status
    with pytest.raises(ValueError, match="boolean"):
        outcome({"status": "scored", "correct": "false"})


def test_pending_and_failed_inference_export_without_false_scores(tmp_path):
    a = make_run(tmp_path / "a", ["one", "two"], [True, True])
    (a / "evaluation.json").unlink()
    rows = read_jsonl(a / "math.jsonl")
    rows[1]["status"] = "error"
    (a / "math.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    exported = export_run(a)
    assert [r["result"] for r in exported] == ["pending", "error"]
    assert all(r["correct"] is None for r in exported)


def test_unfinished_workloads_remain_visible_in_collection_counts(tmp_path):
    a = make_run(tmp_path / "a", ["one"], [True])
    manifest = read_json(a / "manifest.json")
    manifest.update(status="interrupted", expected_records=3)
    manifest["sources"] = {"math": {"selected": 2}, "code": {"selected": 1}}
    write_json(a / "manifest.json", manifest)
    collect_runs([a], tmp_path / "gather")
    rows = {r["workload"]: r for r in csv_rows(tmp_path / "gather/settings.csv")}
    assert rows["math"]["missing"] == "1"
    assert rows["code"]["rows"] == "0" and rows["code"]["missing"] == "1"
    assert rows["code"]["incorrect"] == "0"


def test_gather_rejects_duplicate_settings_and_stale_sources(tmp_path):
    a = make_run(tmp_path / "a", ["one"], [True])
    b = make_run(tmp_path / "b", ["two"], [False])
    with pytest.raises(ValueError, match="Duplicate setting"):
        collect_runs([a, a], tmp_path / "gather")
    collect_runs([a, b], tmp_path / "gather", aliases={str(a): "gpu_model_framework"})
    assert csv_rows(tmp_path / "gather/rows.csv")[0]["setting_id"] == "gpu_model_framework"
    raw = read_jsonl(b / "math.jsonl")[0]
    raw["source_sha256"] = digest("different question")
    (b / "math.jsonl").write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="Source prompt/reference differs"):
        collect_runs([a, b], tmp_path / "gather")


def test_export_refuses_stale_evaluation_and_removes_obsolete_workload_table(tmp_path):
    a = make_run(tmp_path / "a", ["one"], [True])
    export_run(a)
    (a / "tables/code.csv").write_text("obsolete")
    export_run(a)
    assert not (a / "tables/code.csv").exists()
    evaluation = read_json(a / "evaluation.json")
    evaluation["records"][0]["output_sha256"] = digest("old")
    write_json(a / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Stale evaluation"):
        export_run(a)


def test_log_locations_are_not_experimental_confounds(tmp_path):
    a = make_run(tmp_path / "a", ["one"], [True])
    b = make_run(tmp_path / "b", ["two"], [False], gpu="A100")
    for directory in (a, b):
        manifest = read_json(directory / "manifest.json")
        manifest["config"]["launch"] = {"log_path": str(directory / "server.log"), "startup_timeout_seconds": 900}
        write_json(directory / "manifest.json", manifest)
    assert compare_runs(a, b, tmp_path / "pair")["confounds"] == []
