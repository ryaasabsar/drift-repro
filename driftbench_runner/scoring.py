"""Evaluate a saved setting and export its row table on the common host."""
from pathlib import Path

from .comparison import verified_evaluations
from .evaluation import EVALUATOR_VERSION, evaluate_run, load_run
from .tables import export_run

DEFAULT_JUDGE = "Qwen/Qwen3Guard-Gen-0.6B"


def evaluation_complete(run_dir, controls=None):
    manifest, outputs = load_run(run_dir)
    scores = verified_evaluations(run_dir, manifest, outputs)
    complete = bool(outputs) and len(scores) == len(outputs) and all(
        r.get("status") == ("pairwise_only" if key[0] == "chat" else "scored") and r.get("evaluator_version") == EVALUATOR_VERSION
        for key, r in scores.items())
    if complete and controls:
        for key, score in scores.items():
            if key[0] == "safety":
                judge = score.get("judge", {})
                if (judge.get("model") != controls["judge_model"] or judge.get("revision") != controls["judge_revision"] or
                        judge.get("environment", {}).get("device_mode") != controls["judge_device"]):
                    return False
    return complete


def score_run(run_dir, judge_model=DEFAULT_JUDGE, judge_revision=None, judge_device="cpu", labels_output=None, code=True):
    directory = Path(run_dir)
    manifest, outputs = load_run(directory)
    if manifest["status"] != "complete" or len(outputs) != manifest["expected_records"]:
        raise ValueError("Finish this setting's inference before scoring")
    labels = None
    if "safety" in manifest["sources"]:
        from .safety import judge_safety
        labels = Path(labels_output) if labels_output else directory / "safety-labels.jsonl"
        judge_safety(directory, labels, judge_model, judge_revision, judge_device)
    summary = evaluate_run(directory, code, labels)
    export_run(directory)
    return summary
