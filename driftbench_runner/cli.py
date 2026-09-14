import argparse
import json
import os
from pathlib import Path
import time
import traceback
from .common import ROOT, WORKLOADS
from .logging import configure, event


def main():
    parser = argparse.ArgumentParser(description="Run and compare the five published DriftBench workloads")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Launch in the serving framework's Python environment")
    serve.add_argument("--config", required=True)
    serve.add_argument("--dry-run", action="store_true", help="Print argv without loading models or accessing GPUs")
    doctor = sub.add_parser("doctor", help="Inspect accelerators and framework versions on this host")
    doctor.add_argument("--output")
    status = sub.add_parser("status", help="Show saved counts and last recorded activity for a run or suite")
    status.add_argument("run_dir")
    status.add_argument("--json", action="store_true", help="Emit JSON; with --watch, emit one JSON object per update")
    status.add_argument("--watch", action="store_true", help="Keep watching and print changes")
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
    export = sub.add_parser("export", help="Save one setting's per-prompt CSV/JSONL tables")
    export.add_argument("run_dir")
    export.add_argument("--output")
    collect = sub.add_parser("collect", help="Gather settings into tables joined by workload and prompt ID")
    collect.add_argument("run_dirs", nargs="+")
    collect.add_argument("--output", required=True)
    collect.add_argument("--aliases", help="JSON object mapping supplied run paths to unique setting IDs")
    score = sub.add_parser("score", help="Run the safety judge, evaluate all selected workloads, and export tables")
    score.add_argument("run_dir")
    score.add_argument("--judge-model", default="Qwen/Qwen3Guard-Gen-0.6B")
    score.add_argument("--judge-revision")
    score.add_argument("--judge-device", default="cpu", choices=["cpu", "cuda", "auto"])
    score.add_argument("--labels-output")
    score.add_argument("--skip-code", action="store_true")
    suite = sub.add_parser("suite", help="Run model/framework settings sequentially on one accelerator host")
    suite.add_argument("--config", required=True)
    suite.add_argument("--output", required=True)
    suite.add_argument("--settings", nargs="+", help="Only run these setting IDs from the suite")
    suite.add_argument("--limit", type=int, help="First N prompts per workload; default is the suite's selection")
    suite.add_argument("--workloads", nargs="+", choices=list(WORKLOADS))
    suite.add_argument("--device", help="Visible NVIDIA/AMD accelerator indices, e.g. 0")
    for flag in ("resume", "dry-run", "inference-only", "continue-on-error"):
        suite.add_argument("--" + flag, action="store_true")
    for target in (parser, *sub.choices.values()):
        target.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default=argparse.SUPPRESS,
                            help="Console verbosity; detailed events are still saved (default: info)")
        target.add_argument("--progress-interval", type=float, default=argparse.SUPPRESS,
                            help="Seconds between progress updates and long-operation heartbeats (default: 15)")
        target.add_argument("--log-file", default=argparse.SUPPRESS, help="Append readable runner events to this file")
        target.add_argument("--events-file", default=argparse.SUPPRESS, help="Append structured JSONL events to this file")
    args = parser.parse_args()
    interval = getattr(args, "progress_interval", float(os.environ.get("DRIFTBENCH_PROGRESS_INTERVAL", "15")))
    if not 0 < interval < float('inf'):
        parser.error("--progress-interval must be a finite positive number")
    level = getattr(args, "log_level", os.environ.get("DRIFTBENCH_LOG_LEVEL", "info"))
    os.environ["DRIFTBENCH_PROGRESS_INTERVAL"] = str(interval)
    os.environ["DRIFTBENCH_LOG_LEVEL"] = level
    directory = None
    if not getattr(args, "dry_run", False):
        if args.command in ("run", "suite", "compare", "collect"):
            directory = Path(args.output)
        elif args.command in ("evaluate", "judge-safety", "score", "export"):
            directory = Path(args.run_dir)
        elif args.command == "serve":
            from .common import read_json
            try:
                directory = Path(read_json(args.config)["server"]["metadata_path"]).parent
            except (OSError, ValueError, KeyError) as exc:
                parser.error(str(exc))
    child_events = os.environ.get("DRIFTBENCH_EVENTS_FILE")
    default_log = directory / "logs" / f"{args.command}.log" if directory and not child_events else None
    default_events = directory / "logs" / f"{args.command}.events.jsonl" if directory else None
    try:
        configure(level, getattr(args, "log_file", default_log),
                  getattr(args, "events_file", child_events or default_events), interval)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    started = time.monotonic()
    event("Command started", stage=args.command, output=str(directory) if directory else None)
    try:
        dispatch(args)
    except KeyboardInterrupt:
        event("Interrupted by user", stage=args.command, level="warning")
        raise SystemExit(130)
    except Exception as exc:
        event(str(exc), stage=args.command, level="error", error_type=type(exc).__name__)
        event("Exception details", stage=args.command, level="debug", traceback=traceback.format_exc())
        raise SystemExit(1) from None
    event("Command finished", stage=args.command, elapsed_seconds=round(time.monotonic() - started, 1))


def dispatch(args):
    if args.command == "status":
        from .status import show_status
        from .logging import progress_interval
        show_status(args.run_dir, args.json, args.watch, progress_interval())
    elif args.command == "serve":
        from .serving import serve
        serve(args.config, args.dry_run)
    elif args.command == "doctor":
        from .hardware import discover
        from .common import write_json
        result = discover()
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command in ("run", "preflight"):
        from .inference import run
        result = run(args.config, getattr(args, "output", ""), args.workloads, args.limit,
                     getattr(args, "resume", False), args.command == "preflight")
        if args.command == "preflight":
            print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate_run
        from .tables import export_run
        evaluate_run(args.run_dir, args.code, args.safety_labels)
        export_run(args.run_dir)
    elif args.command == "compare":
        from .comparison import compare_runs
        compare_runs(args.baseline, args.candidate, args.output, args.semantic, args.allow_confounded)
    elif args.command == "judge-safety":
        from .safety import judge_safety
        judge_safety(args.run_dir, args.output, args.model, args.revision, args.device)
    elif args.command == "export":
        from .tables import export_run
        rows = export_run(args.run_dir, args.output)
        event("Responses exported", stage="export", completed=len(rows), output=args.output or str(Path(args.run_dir) / "tables"))
    elif args.command == "collect":
        from .common import read_json
        from .tables import collect_runs
        collect_runs(args.run_dirs, args.output, read_json(args.aliases) if args.aliases else None)
    elif args.command == "score":
        from .scoring import score_run, evaluation_complete
        score_run(args.run_dir, args.judge_model, args.judge_revision, args.judge_device, args.labels_output, not args.skip_code)
        if not args.skip_code and not evaluation_complete(args.run_dir):
            raise RuntimeError("Some evaluations remain pending; inspect evaluation.json")
    elif args.command == "suite":
        from .suite import run_suite
        result = run_suite(args.config, args.output, args.settings, args.limit, args.workloads, args.device,
                           args.resume, args.dry_run, args.inference_only, args.continue_on_error)
        if args.dry_run:
            print(json.dumps(result, indent=2))
        else:
            event("Suite finished", stage="suite", status=result['status'],
                  level="error" if result['status'] == "failed" else "info",
                  completed_settings=sum(e["status"] == "complete" for e in result["settings"].values()),
                  total_settings=len(result["settings"]), selected_settings=len(result["selected"]),
                  tables=f"{args.output}/collection")
            if result["status"] == "failed":
                raise SystemExit(1)


if __name__ == "__main__":
    main()
