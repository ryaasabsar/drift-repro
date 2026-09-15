"""Evaluate a saved setting and export its row table on the common host."""
from .comparison import verified_evaluations
from .evaluation import EVALUATOR_VERSION, load_run

DEFAULT_JUDGE = "meta-llama/Llama-Guard-3-8B"


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
