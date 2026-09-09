import argparse
import json
from .common import ROOT, WORKLOADS


def main():
    parser = argparse.ArgumentParser(description="Run and compare the five published DriftBench workloads")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default=str(ROOT / "configs/rtx3060-qwen35-vllm.json"))
        cmd.add_argument("--workloads", nargs="+", choices=list(WORKLOADS), default=list(WORKLOADS))
        cmd.add_argument("--limit", type=int, help="First N per workload; omitted means complete published files")
        if name == "run":
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("run_dir")
    evaluate.add_argument("--code", action="store_true", help="Execute HumanEval completions with the isolated worker")
    evaluate.add_argument("--safety-labels", help="JSONL labels produced by judge-safety or imported annotations")
    compare = sub.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--output", required=True)
    compare.add_argument("--semantic", action="store_true", help="Compute chat embedding cosine drift on CPU")
    compare.add_argument("--allow-confounded", action="store_true", help="Label and allow multiple changed setup factors")
    judge = sub.add_parser("judge-safety")
    judge.add_argument("run_dir")
    judge.add_argument("--model", default="Qwen/Qwen3Guard-Gen-0.6B")
    judge.add_argument("--revision", help="Defaults to the pinned Qwen3Guard revision; main for other judges")
    judge.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    judge.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in ("run", "preflight"):
        from .inference import run
        result = run(args.config, getattr(args, "output", ""), args.workloads, args.limit,
                     getattr(args, "resume", False), args.command == "preflight")
        if args.command == "preflight":
            print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate_run
        evaluate_run(args.run_dir, args.code, args.safety_labels)
    elif args.command == "compare":
        from .comparison import compare_runs
        compare_runs(args.baseline, args.candidate, args.output, args.semantic, args.allow_confounded)
    elif args.command == "judge-safety":
        from .safety import judge_safety
        judge_safety(args.run_dir, args.output, args.model, args.revision, args.device)


if __name__ == "__main__":
    main()
