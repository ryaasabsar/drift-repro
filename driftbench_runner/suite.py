"""Run selected model/framework settings sequentially on one accelerator host."""
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .common import ROOT, WORKLOADS, digest, keyed, load_workloads, now, read_json, write_json
from .evaluation import load_run
from .hardware import validate_target
from .scoring import DEFAULT_JUDGE
from .serving import launch_command
from .tables import collect_runs, export_run
from .logging import EventFollower, activity, event, progress_interval


def load_plan(config_path, output_dir, limit=None, workloads=None, device=None, selected=None, frameworks=None):
    path = Path(config_path).resolve()
    spec = read_json(path)
    unknown = set(spec) - {"suite_id", "settings", "workloads", "limit", "evaluation", "comparisons"}
    if unknown:
        raise ValueError(f"Unknown suite fields: {sorted(unknown)}")
    selected_workloads = workloads or spec.get("workloads", list(WORKLOADS))
    if not selected_workloads or len(set(selected_workloads)) != len(selected_workloads) or set(selected_workloads) - set(WORKLOADS):
        raise ValueError("Choose unique, known workloads")
    limit = spec.get("limit") if limit is None else limit
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or null for full workloads")
    _, sources = load_workloads(selected_workloads, limit)
    out = Path(output_dir).resolve()
    settings, seen, vendors = [], set(), set()
    for item in spec["settings"]:
        if set(item) - {"id", "config", "server_python"}:
            raise ValueError(f"Unknown setting fields: {item}")
        sid = item["id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", sid) or sid in seen:
            raise ValueError(f"Setting IDs must be unique safe directory names: {sid!r}")
        seen.add(sid)
        config = read_json(path.parent / item["config"])
        validate_target(config)
        vendor = config.get("hardware", {}).get("vendor", "nvidia")
        vendors.add(vendor)
        if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
            raise ValueError(f"Pin {sid}'s model revision before running a suite")
        run_dir = out / "settings" / sid
        config["setup_id"] = sid
        if device is not None:
            if vendor == "tenstorrent":
                raise ValueError("Select Tenstorrent chips in the profile's tt_visible_devices/mesh settings")
            if not re.fullmatch(r"\d+(,\d+)*", device):
                raise ValueError("--device must be comma-separated accelerator indices, e.g. 0")
            key = "CUDA_VISIBLE_DEVICES" if vendor == "nvidia" else "HIP_VISIBLE_DEVICES"
            env = config.setdefault("launch", {}).setdefault("env", {})
            if key in env and env[key] != device:
                raise ValueError(f"{sid}'s device selection conflicts with --device")
            env[key] = device
        if config.get("transport", "offline") == "http":
            config["server"]["metadata_path"] = str(run_dir / "server.json")
            config.setdefault("launch", {})["log_path"] = str(run_dir / "server.log")
            launch_command(config)  # Validate mapped flags without importing a framework.
        server_python = (path.parent / item["server_python"]).absolute()
        # Do not resolve the Python symlink: that would escape its virtualenv.
        settings.append({"id": sid, "config": config, "server_python": str(server_python),
                         "config_path": str(run_dir / "config.json"), "run_dir": str(run_dir)})
    if not settings or len(vendors) != 1:
        raise ValueError("A suite must have settings for one accelerator vendor on this host")
    evaluation = {"judge_model": DEFAULT_JUDGE, "judge_revision": None, "judge_device": "cuda", "code": True,
                  **spec.get("evaluation", {})}
    if set(evaluation) - {"judge_model", "judge_revision", "judge_device", "code"}:
        raise ValueError("Unknown evaluation settings")
    if evaluation["judge_device"] not in ("cpu", "cuda", "auto") or type(evaluation["code"]) is not bool:
        raise ValueError("Invalid evaluation device or code setting")
    pinned_judge = read_json(Path(__file__).resolve().parents[1] / "models.lock.json")["safety"]
    if evaluation["judge_revision"] is None and evaluation["judge_model"] == pinned_judge["model"]:
        evaluation["judge_revision"] = pinned_judge["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", evaluation["judge_revision"] or ""):
        raise ValueError("Pin the suite's safety judge revision to a 40-character Hub commit")
    comparisons = spec.get("comparisons", [])
    for pair in comparisons:
        if set(pair) - {"baseline", "candidate", "semantic", "allow_confounded"}:
            raise ValueError("Unknown comparison fields")
        if pair["baseline"] not in seen or pair["candidate"] not in seen or pair["baseline"] == pair["candidate"]:
            raise ValueError("Comparison IDs must name two different settings in the suite")
    if selected is not None:
        unknown = set(selected) - seen
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        settings = [item for item in settings if item['id'] in selected]
    if frameworks is not None:
        unknown = set(frameworks) - {item['config']['backend'] for item in settings}
        if unknown:
            raise ValueError(f"Frameworks unavailable in the selected suite: {sorted(unknown)}")
        settings = [item for item in settings if item['config']['backend'] in frameworks]
    if not settings:
        raise ValueError('No settings match the requested filters')
    selected_ids = {item['id'] for item in settings}
    comparisons = [pair for pair in comparisons if {pair['baseline'], pair['candidate']} <= selected_ids]
    return {"suite_id": spec["suite_id"], "workloads": selected_workloads, "limit": limit,
            "sources": sources, "expected_records_per_setting": sum(s["selected"] for s in sources.values()),
            "device": device, "evaluation": evaluation, "settings": settings, "comparisons": comparisons}


def inference_complete(item, plan):
    path = Path(item["run_dir"]) / "manifest.json"
    if not path.exists():
        return False
    manifest, outputs = load_run(item["run_dir"])
    if manifest["config"] != item["config"] or manifest["sources"] != plan["sources"]:
        raise ValueError(f"{item['id']}: saved config or input selection differs; choose a new suite output directory")
    expected_rows, _ = load_workloads(plan["workloads"], plan["limit"])
    expected = keyed(expected_rows)
    for key, row in outputs.items():
        if key not in expected or row["source_sha256"] != expected[key]["source_sha256"] or row["prompt"] != expected[key]["prompt"]:
            raise ValueError(f"Saved response has an invalid source: {key}")
        if row["request_sha256"] != digest({"input_ids": row["input_token_ids"], "prompt": row["rendered_prompt"]}):
            raise ValueError(f"Saved request trace changed: {key}")
        if row["status"] != "ok":
            raise ValueError(f"Saved inference failed for {key}; inspect the setting before resuming")
    complete = manifest["status"] == "complete" and set(outputs) == set(expected)
    if complete:
        identity = {"config": item["config"], "sources": plan["sources"],
                    "requests": [[*key, outputs[key]["request_sha256"], outputs[key]["source_sha256"]] for key in expected]}
        if item["config"].get("transport", "offline") == "http":
            identity["server"] = manifest["server_provenance"]["serving_fingerprint"]
        if manifest["expected_records"] != len(expected) or digest(identity) != manifest["run_fingerprint"]:
            raise ValueError("Saved run fingerprint does not match its response records")
    return complete


def stop_process(process, launcher=False):
    if process.poll() is not None:
        return
    if launcher:
        process.terminate()  # The launcher owns and shuts down its server's group.
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        # Do not start another accelerator job after uncertain server cleanup.
        raise RuntimeError("Process did not stop cleanly; inspect its log before continuing")


def run_command(command, log_path, environment, followers=()):
    log_path = Path(log_path)
    events_path = log_path.parent / "logs" / (log_path.stem + ".events.jsonl")
    follower = EventFollower(events_path)
    environment = {**environment, "PYTHONUNBUFFERED": "1", "DRIFTBENCH_EVENTS_FILE": str(events_path),
                   "DRIFTBENCH_LOG_STAGE": log_path.stem}
    event("Process started", stage=log_path.stem, setting=environment.get("DRIFTBENCH_LOG_SETTING"), log=str(log_path))
    with Path(log_path).open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            started = last_notice = time.monotonic()
            last_child_event = None
            while process.poll() is None:
                for additional in followers:
                    additional.drain()
                follower.drain()
                if follower.last_event is not last_child_event:
                    last_child_event = follower.last_event
                    last_notice = time.monotonic()
                if time.monotonic() - last_notice >= max(1, progress_interval() * 2):
                    event("Waiting for child process update", stage=log_path.stem,
                          setting=environment.get("DRIFTBENCH_LOG_SETTING"),
                          elapsed_seconds=round(time.monotonic() - started, 1), log=str(log_path),
                          last_activity=last_child_event["message"] if last_child_event else "process starting",
                          **{k: v for k, v in (last_child_event or {}).items()
                             if k in ("workload", "completed", "total", "prompt_id")})
                    last_notice = time.monotonic()
                time.sleep(0.2)
            follower.drain()
            if process.returncode:
                raise RuntimeError(f"Command exited {process.returncode}; see {log_path}")
        finally:
            stop_process(process)
            for additional in followers:
                additional.drain()
            follower.drain()


def run_inference(item, plan, resume, environment):
    run_dir = Path(item["run_dir"])
    command = [sys.executable, "-m", "driftbench_runner", "run", "--config", item["config_path"],
               "--output", str(run_dir), "--workloads", *plan["workloads"]]
    if plan["limit"] is not None:
        command += ["--limit", str(plan["limit"])]
    if resume and (run_dir / "manifest.json").exists():
        command.append("--resume")
    with (run_dir / "launcher.log").open("a") as log:
        started = time.time()
        events_path = run_dir / "logs/launcher.events.jsonl"
        follower = EventFollower(events_path)
        launcher_env = {**environment, "PYTHONUNBUFFERED": "1", "DRIFTBENCH_EVENTS_FILE": str(events_path)}
        event("Starting serving framework", stage="startup", setting=item["id"], model=item["config"]["model"],
              framework=item["config"]["backend"], log=str(run_dir / "server.log"))
        server = subprocess.Popen([item["server_python"], "-m", "driftbench_runner", "serve", "--config", item["config_path"]],
                                  cwd=ROOT, env=launcher_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            config = item["config"]
            deadline = time.monotonic() + config.get("launch", {}).get("startup_timeout_seconds", 900) + config["server"].get("timeout_seconds", 900)
            while True:
                follower.drain()
                if server.poll() is not None:
                    raise RuntimeError(f"{item['id']}: server exited {server.returncode}; see {run_dir / 'launcher.log'}")
                metadata_path = Path(config["server"]["metadata_path"])
                if metadata_path.exists():
                    metadata = read_json(metadata_path)
                    fresh = datetime.fromisoformat(metadata["created_at"]).timestamp() >= started
                    if fresh and metadata.get("launcher_pid") == server.pid and metadata["status"] == "ready":
                        break
                    if fresh and metadata["status"] == "failed":
                        raise RuntimeError(metadata.get("error", "Server failed"))
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{item['id']}: server startup timed out")
                time.sleep(1)
            follower.drain()
            run_command(command, run_dir / "inference.log", environment, (follower,))
        finally:
            with activity("Stopping server and releasing accelerator", stage="shutdown", setting=item["id"]):
                stop_process(server, launcher=True)
                follower.drain()
            metadata_path = Path(item["config"]["server"]["metadata_path"])
            if metadata_path.exists():
                final = read_json(metadata_path)
                if final.get("launcher_pid") == server.pid and "stopped_at" not in final:
                    raise RuntimeError("Server did not stop cleanly; inspect launcher.log before continuing")


def run_suite(config_path, output_dir, selected=None, limit=None, workloads=None, device=None,
              resume=False, dry_run=False, inference_only=True, continue_on_error=False, frameworks=None):
    plan = load_plan(config_path, output_dir, limit, workloads, device, selected, frameworks)
    ids = {item["id"] for item in plan["settings"]}
    selected = ids
    if dry_run:
        return {**plan, "selected": sorted(selected), "server_python_available": {
            item["id"]: Path(item["server_python"]).is_file() for item in plan["settings"]}}
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / ".suite.lock").open("w") as lock, (ROOT / "results/.gpu-experiment.lock").open("w") as gpu_lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another GPU experiment or suite is already running in this workspace") from exc
            return _run_suite_locked(plan, out, selected, resume, inference_only, continue_on_error)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def _run_suite_locked(plan, out, selected, resume, inference_only, continue_on_error):
    state_path = out / "suite.json"
    fingerprint = digest(plan)
    if state_path.exists():
        state = read_json(state_path)
        if not resume:
            raise ValueError("Suite exists; use --resume or a new output directory")
        if state["plan_fingerprint"] != fingerprint:
            raise ValueError("Suite settings, inputs, or evaluation controls changed; use a new output directory")
    else:
        if any((out / "settings").glob("*/manifest.json")):
            raise ValueError("Saved runs exist without a suite manifest")
        state = {"schema_version": 1, "suite_id": plan["suite_id"], "plan_fingerprint": fingerprint, "plan": plan,
                 "created_at": now(), "settings": {item["id"]: {"status": "pending"} for item in plan["settings"]}}
    state.pop("error", None)
    state.pop("collection_error", None)
    state.update(status="running", updated_at=now(), selected=sorted(selected))
    write_json(state_path, state)
    event("Suite started", stage="suite", suite=plan["suite_id"], selected_settings=len(selected),
          prompts_per_setting=plan["expected_records_per_setting"], workloads=", ".join(plan["workloads"]), resume=resume)
    failed = False
    try:
        selected_index = 0
        for item in plan["settings"]:
            sid = item["id"]
            if sid not in selected:
                continue
            selected_index += 1
            event("Setting started", stage="suite", setting=sid, position=f"{selected_index}/{len(selected)}",
                  model=item["config"]["model"], framework=item["config"]["backend"])
            entry = state["settings"][sid]
            directory = Path(item["run_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            entry.pop("error", None)
            environment = dict(os.environ)
            environment["DRIFTBENCH_LOG_SETTING"] = sid
            environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
            if plan["device"] is not None:
                vendor = item["config"].get("hardware", {}).get("vendor", "nvidia")
                environment["CUDA_VISIBLE_DEVICES" if vendor == "nvidia" else "HIP_VISIBLE_DEVICES"] = plan["device"]
            try:
                complete = inference_complete(item, plan)
                if not complete:
                    if not os.access(item["server_python"], os.X_OK):
                        raise ValueError(f"Missing serving Python: {item['server_python']}")
                    write_json(item["config_path"], item["config"])
                    entry.update(status="inference_running", started_at=now())
                    write_json(state_path, state)
                    event("Inference required", stage="inference", setting=sid, total=plan['expected_records_per_setting'])
                    run_inference(item, plan, resume, environment)
                    if not inference_complete(item, plan):
                        raise RuntimeError("Inference command returned without a complete run")
                else:
                    event("Reusing completed inference", stage="resume", setting=sid, completed=plan['expected_records_per_setting'])
                event("Evaluation deferred to the safety/code/final stages", stage="evaluation", setting=sid)
                export_run(directory)
                entry.update(status="inference_complete", updated_at=now())
                event("Setting finished", stage="suite", setting=sid, status=entry['status'],
                      rows=str(directory / 'tables/rows.csv'))
            except Exception as exc:
                failed = True
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
                event("Setting failed", stage="suite", setting=sid, level="error", error=str(exc), logs=str(directory))
                if not continue_on_error or "did not stop cleanly" in str(exc):
                    raise
            except KeyboardInterrupt:
                entry.update(status="interrupted", updated_at=now())
                raise
            finally:
                write_json(state_path, state)
        states = [e["status"] for e in state["settings"].values()]
        state["status"] = "failed" if failed else (
            "inference_complete" if all(s == "inference_complete" for s in states) else "partial")
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        available = [item["run_dir"] for item in plan["settings"] if (Path(item["run_dir"]) / "manifest.json").exists()]
        if available:
            try:
                collect_runs(available, out / "collection")
            except Exception as exc:
                state.update(status="failed", collection_error=str(exc))
                event("Could not gather result tables", stage="collection", level="error", error=str(exc))
        state["updated_at"] = now()
        write_json(state_path, state)
    return state
