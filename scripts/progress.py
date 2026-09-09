"""Read only: summarize the currently saved records of a run."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    directory = Path(args.run_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = {"run": str(directory), "inference_status": manifest["status"], "workloads": {}}
    for workload, source in manifest["sources"].items():
        path = directory / f"{workload}.jsonl"
        saved = sum(1 for line in path.open() if line.endswith("\n")) if path.exists() else 0
        report["workloads"][workload] = {"saved": saved, "expected": source["selected"]}
    labels = directory / "safety-labels.jsonl"
    report["safety_judgments"] = sum(1 for line in labels.open() if line.endswith("\n")) if labels.exists() else 0
    evaluation = directory / "evaluation.json"
    if evaluation.exists():
        report["evaluation"] = json.loads(evaluation.read_text())["summary"]["workloads"]
        report["evaluation_matches_saved_counts"] = all(
            report["evaluation"].get(workload, {}).get("generated") == counts["saved"]
            for workload, counts in report["workloads"].items()
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
