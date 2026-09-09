import ast
import collections
import csv
import math
import platform
import re
import resource
import shutil
import string
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from functools import lru_cache

from .common import ROOT, WORKLOADS, digest, keyed, load_workloads, read_json, read_jsonl, write_json

EVALUATOR_VERSION = "driftbench-runner-eval-v3"


def final_text(text):
    # Reasoning is retained in raw outputs; only the final answer is scored.
    return text.rsplit("</think>", 1)[-1].strip()


def extract_number(text):
    text = final_text(text).replace(",", "")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    elif "####" in text:
        text = text.rsplit("####", 1)[-1]
    matches = re.findall(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", text)
    if not matches:
        return None
    try:
        parts = matches[-1].split("/")
        value = Decimal(parts[0].strip())
        if len(parts) == 2:
            value /= Decimal(parts[1].strip())
        return str(value.normalize()) if value.is_finite() else None
    except (InvalidOperation, ZeroDivisionError):
        return None


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_f1(prediction, reference):
    prediction, reference = normalize_answer(prediction), normalize_answer(reference)
    # Match LongBench's special treatment of yes/no/unanswerable answers.
    special = {"yes", "no", "unanswerable"}
    if (prediction in special or reference in special) and prediction != reference:
        return 0.0
    pred, gold = prediction.split(), reference.split()
    if not pred or not gold:
        return float(pred == gold)
    common = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    return 2 * common / (len(pred) + len(gold)) if common else 0.0


def code_units(prompt, output, entry_point):
    text = final_text(output)
    # An unfinished Markdown fence does not make an otherwise complete Python
    # function incorrect. Never repair or complete the Python code itself.
    blocks = [body for language, body in re.findall(r"```([^\n]*)\n(.*?)(?:```|$)", text, re.DOTALL)
              if language.strip().lower() in ("", "python", "py")]
    if blocks:
        text = next((b for b in blocks if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", b)), blocks[0])
    if re.search(rf"(?m)^def\s+{re.escape(entry_point)}\s*\(", text):
        # The prompt may supply helper functions, not just imports. Execute the
        # answer separately in the same namespace, overriding the target stub.
        # Separate compilation also preserves valid __future__ imports.
        return [prompt, text]
    # Preserve leading indentation for raw HumanEval function-body completions.
    if not blocks and "</think>" not in output:
        text = output
    return [prompt + text]


def code_candidate(prompt, output, entry_point):
    return "\n".join(code_units(prompt, output, entry_point))


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 ** 2, 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def bubblewrap():
    local = ROOT / ".tools/bubblewrap-root/usr/bin/bwrap"
    return str(local) if local.exists() else shutil.which("bwrap")


@lru_cache(maxsize=1)
def code_environment():
    cpu_name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_name = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                         if line.startswith("model name")), cpu_name)
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu_name,
            "sandbox": "bubblewrap", "test_seed": 42, "python_hash_seed": 42,
            "cpu_seconds": 5, "wall_seconds": 10, "address_space_bytes": 1024 ** 3}


def execute_code(row, output):
    tool = bubblewrap()
    if not tool:
        return {"status": "pending", "reason": "bubblewrap is required for isolated HumanEval execution"}
    units = code_units(row["prompt"], output, row["entry_point"])
    candidate_hash = digest(units)
    try:
        for unit in units:
            ast.parse(unit)
    except SyntaxError as exc:
        return {"status": "scored", "correct": False, "score": 0.0,
                "detail": f"syntax_error: {exc.msg}", "candidate_sha256": candidate_hash}
    # Mount only the Python runtime, system libraries and one test case. The
    # workspace, home directory, credentials and network are absent.
    runtime = Path(sys.base_prefix).resolve()
    python_rel = Path(sys.executable).resolve().relative_to(runtime)
    with tempfile.TemporaryDirectory(prefix="driftbench-humaneval-") as directory:
        path = Path(directory)
        test_unit = row["test_cases"] + f"\ncheck({row['entry_point']})\n"
        # Match HumanEval's separate exec namespace: generated __main__ examples
        # are not run as the benchmark's test suite.
        program = "import random\nrandom.seed(42)\nnamespace = {}\n"
        for index, unit in enumerate(units + [test_unit]):
            program += f"exec(compile({unit!r}, '<humaneval-unit-{index}>', 'exec'), namespace)\n"
        program += "print('DRIFTBENCH_TEST_PASSED')\n"
        (path / "main.py").write_text(program)
        command = [tool, "--unshare-all", "--die-with-parent", "--new-session",
                   "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"]
        if Path("/lib64").exists():
            command += ["--ro-bind", "/lib64", "/lib64"]
        command += ["--ro-bind", str(runtime), "/python", "--proc", "/proc", "--dev", "/dev",
                    "--tmpfs", "/tmp", "--chdir", "/tmp", "--ro-bind", directory, "/case",
                    "--setenv", "PATH", "/python/bin:/usr/bin", "--setenv", "PYTHONHASHSEED", "42", "--",
                    "/python/" + str(python_rel), "-s", "-S", "/case/main.py"]
        with (path / "stdout").open("w+") as stdout, (path / "stderr").open("w+") as stderr:
            try:
                proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=10, preexec_fn=_limits,
                                      env={"PATH": "/usr/bin:/bin"})
                stdout.seek(0)
                stderr.seek(0)
                output_text, error = stdout.read(4096), stderr.read(4096)
            except subprocess.TimeoutExpired:
                return {"status": "scored", "correct": False, "score": 0.0, "detail": "timeout",
                        "candidate_sha256": candidate_hash}
        if "bwrap:" in error:
            return {"status": "pending", "reason": error.strip()}
        passed = proc.returncode == 0 and "DRIFTBENCH_TEST_PASSED" in output_text
        return {"status": "scored", "correct": passed, "score": float(passed),
                "detail": "passed" if passed else error[-2000:] or f"exit {proc.returncode}",
                "candidate_sha256": candidate_hash}


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p, den = successes / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    outputs = keyed([r for workload in manifest["sources"] for r in read_jsonl(run_dir / f"{workload}.jsonl")])
    return manifest, outputs


def evaluate_run(run_dir, execute=False, safety_labels=None):
    run_dir = Path(run_dir)
    manifest, outputs = load_run(run_dir)
    sources, source_info = load_workloads(list(manifest["sources"]))
    source_map = keyed(sources)
    for name, info in manifest["sources"].items():
        if source_info[name]["sha256"] != info["sha256"]:
            raise ValueError(f"Dataset content changed: {name}")
    labels = keyed(read_jsonl(safety_labels)) if safety_labels else {}
    scores = []
    for key, output in outputs.items():
        row = source_map[key]
        if output["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"Source mismatch: {key}")
        result = {"workload": key[0], "prompt_id": key[1], "status": "pending",
                  "output_sha256": digest(output["output_text"]), "source_sha256": output["source_sha256"],
                  "evaluator_version": EVALUATOR_VERSION}
        text = final_text(output["output_text"])
        if output["status"] != "ok":
            result.update(status="error", reason="inference failed")
        elif key[0] == "math":
            answer = extract_number(text)
            correct = answer is not None and answer == extract_number(row["answer"])
            result.update(status="scored", extracted_answer=answer, reference=row["answer"],
                          correct=correct, score=float(correct), method="numeric_exact_match")
        elif key[0] == "long_context":
            score = max(qa_f1(text, answer) for answer in row["answers"])
            result.update(status="scored", score=score, correct=score >= 0.5,
                          reference=row["answers"], method="longbench_token_f1", threshold=0.5)
        elif key[0] == "code":
            result.update(execute_code(row, output["output_text"]) if execute else
                          {"reason": "Use --code to run HumanEval in bubblewrap"})
            result["method"] = "humaneval_execution_pass_at_1"
            result["execution_environment"] = code_environment()
        elif key[0] == "safety":
            label = labels.get(key)
            if label:
                if label.get("output_sha256") != result["output_sha256"] or label.get("source_sha256") != output["source_sha256"]:
                    raise ValueError(f"Safety annotation belongs to different prompt/output: {key}")
                if label.get("label") not in ("safe", "unsafe") or not label.get("judge"):
                    raise ValueError(f"Invalid safety label or missing judge provenance: {key}")
                result.update(status="scored", label=label["label"], correct=label["label"] == "safe",
                              severity=label.get("severity", label["label"]),
                              score=float(label["label"] == "safe"), method="safety_classification", judge=label["judge"])
            else:
                result["reason"] = "Requires labels from judge-safety; no heuristic substitute is used"
        else:
            result.update(status="pairwise_only", reason="Chat semantic drift requires a second run")
        scores.append(result)
    summary = {"schema_version": 1, "run_fingerprint": manifest["run_fingerprint"],
               "evaluator_version": EVALUATOR_VERSION, "workloads": {}}
    for workload, source in manifest["sources"].items():
        rows = [r for r in scores if r["workload"] == workload]
        scored = [r for r in rows if r["status"] == "scored"]
        generated = [o for (w, _), o in outputs.items() if w == workload]
        correct = sum(r["correct"] for r in scored)
        latencies = [r["latency_seconds"] for r in generated if r.get("latency_seconds") is not None]
        info = {"expected": source["selected"], "generated": len(generated),
                "missing": source["selected"] - len(generated), "scored": len(scored),
                "pending": sum(r["status"] == "pending" for r in rows),
                "pairwise_only": sum(r["status"] == "pairwise_only" for r in rows),
                "correct": correct if scored else None,
                "accuracy": correct / len(scored) if scored else None,
                "accuracy_wilson95": wilson(correct, len(scored)),
                "mean_score": sum(r["score"] for r in scored) / len(scored) if scored else None,
                "length_limited": sum(r["finish_reason"] == "length" for r in generated),
                "input_tokens": sum(r["input_tokens"] for r in generated),
                "output_tokens": sum(r["output_tokens"] for r in generated),
                "controversial": sum(r.get("severity") == "controversial" for r in scored),
                "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None}
        summary["workloads"][workload] = info
    write_json(run_dir / "evaluation.json", {"summary": summary, "records": scores})
    with (run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["workload", *next(iter(summary["workloads"].values())).keys()])
        writer.writeheader()
        writer.writerows({"workload": name, **info} for name, info in summary["workloads"].items())
    print(f"Saved {run_dir / 'evaluation.json'} and summary.csv", flush=True)
    return summary
