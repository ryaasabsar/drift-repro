"""Week 1: repeatable cross-device generation, scoring and comparison.

    generate  send the frozen cases to a running vLLM server (stdlib only)
    score     grade outputs with the parent repo's DriftBench evaluator
    report    accuracy, repeatability, cross-device flips, first divergences
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "data" / "cases.jsonl"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# Same sampling controls as the original DriftBench configs.
SAMPLING = {"max_tokens": 512, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42, "n": 1,
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "min_p": 0.0}
DEVICE_ORDER = ["a100", "mi210", "p150b"]  # NVIDIA first: it is the reference side of every pair


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def http(url, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


# ---------------------------------------------------------------- generate

def generate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    served = [m["id"] for m in http(base + "/v1/models")["data"]]
    if MODEL not in served:
        sys.exit(f"server does not serve {MODEL}: {served}")
    cases = read_jsonl(CASES)
    if args.limit:  # smoke test: the first N cases of each workload
        cases = [c for w in ("code", "math") for c in [c for c in cases if c["workload"] == w][:args.limit]]
    meta_path = out / "meta.json"
    if not meta_path.exists():
        from importlib import metadata
        packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
        meta = {"device": args.device, "repeat": args.repeat, "model": MODEL, "cases_sha256": sha256(CASES),
                "sampling": SAMPLING, "logprobs": args.logprobs, "token_ids": not args.no_token_ids,
                "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
                "python": sys.version.split()[0], "packages": dict(sorted(packages.items())),
                "env": {k: v for k, v in os.environ.items() if k.endswith("VISIBLE_DEVICES") or k.startswith(
                    ("VLLM_", "HIP_", "ROCR_", "CUBLAS_", "MESH_", "ARCH_", "TT_", "PYTHONHASHSEED"))}}
        meta_path.write_text(json.dumps(meta, indent=1))
    elif json.loads(meta_path.read_text())["cases_sha256"] != sha256(CASES):
        sys.exit("cases.jsonl changed since this run started; use a new output folder")
    done_path = out / "outputs.jsonl"
    done = {r["prompt_id"] for r in read_jsonl(done_path)} if done_path.exists() else set()
    todo = [c for c in cases if c["prompt_id"] not in done]
    print(f"[{args.device} r{args.repeat}] {len(done)} done, {len(todo)} to generate", flush=True)
    for i, case in enumerate(todo, 1):
        # One request at a time: together with --max-num-seqs 1 this is batch size 1.
        payload = {"model": MODEL, "prompt": case["input_ids"], "add_special_tokens": False,
                   "stream": False, **SAMPLING}
        if not args.no_token_ids:
            payload["return_token_ids"] = True
        if args.logprobs:
            payload["logprobs"] = args.logprobs
        start = time.time()
        raw = http(base + "/v1/completions", payload)
        choice = raw["choices"][0]
        if raw["usage"]["prompt_tokens"] != len(case["input_ids"]):
            sys.exit(f"{case['prompt_id']}: server changed the input length")
        if choice["finish_reason"] not in ("stop", "length"):
            sys.exit(f"{case['prompt_id']}: abnormal finish {choice['finish_reason']}")
        top = None
        if choice.get("logprobs") and choice["logprobs"].get("top_logprobs"):
            # Per generated position: the top-k [token, logprob] pairs, best first.
            top = [sorted(d.items(), key=lambda kv: -kv[1]) for d in choice["logprobs"]["top_logprobs"]]
        append_jsonl(done_path, {
            "prompt_id": case["prompt_id"], "workload": case["workload"], "text": choice["text"],
            "token_ids": choice.get("token_ids"), "n_tokens": raw["usage"]["completion_tokens"],
            "finish_reason": choice["finish_reason"], "top_logprobs": top,
            "seconds": round(time.time() - start, 3)})
        if i % 25 == 0 or i == len(todo):
            print(f"[{args.device} r{args.repeat}] {len(done) + i}/{len(cases)}", flush=True)


# ---------------------------------------------------------------- score

def score(args):
    # Reuse DriftBench's exact math extraction and sandboxed HumanEval execution.
    sys.path.insert(0, str(HERE.parent))
    from driftbench_runner.evaluation import execute_code, extract_number, final_text
    cases = {c["prompt_id"]: c for c in read_jsonl(CASES)}
    for run in args.runs:
        run = Path(run)
        scores_path = run / "scores.jsonl"
        scored = {r["prompt_id"] for r in read_jsonl(scores_path)} if scores_path.exists() else set()
        todo = [o for o in read_jsonl(run / "outputs.jsonl") if o["prompt_id"] not in scored]
        for o in todo:
            case = cases[o["prompt_id"]]
            if case["workload"] == "math":
                got = extract_number(final_text(o["text"]))
                correct, detail = got is not None and got == extract_number(case["answer"]), got
            else:
                result = execute_code(case, o["text"])
                if result["status"] != "scored":
                    sys.exit(f"cannot score code: {result.get('reason')} (install bubblewrap)")
                correct, detail = result["correct"], result["detail"][:200]
            append_jsonl(scores_path, {"prompt_id": o["prompt_id"], "workload": case["workload"],
                                       "correct": bool(correct), "detail": detail})
        print(f"{run}: scored {len(todo)} new outputs")


# ---------------------------------------------------------------- report

def load_runs(root):
    """root/<device>/r<k>/outputs.jsonl -> {device: {repeat: {prompt_id: output}}}."""
    runs = defaultdict(dict)
    for path in sorted(Path(root).glob("*/r*/outputs.jsonl")):
        device, repeat = path.parent.parent.name, int(path.parent.name[1:])
        rows = {r["prompt_id"]: r for r in read_jsonl(path)}
        scores = path.parent / "scores.jsonl"
        if scores.exists():
            for s in read_jsonl(scores):
                rows[s["prompt_id"]]["correct"] = s["correct"]
        runs[device][repeat] = rows
    return runs


def sequences(a, b):
    # Compare token IDs when both servers returned them; otherwise fall back to text.
    if a.get("token_ids") is not None and b.get("token_ids") is not None:
        return a["token_ids"], b["token_ids"], "token"
    return a["text"], b["text"], "char"


def same(a, b):
    sa, sb, _ = sequences(a, b)
    return sa == sb and a["finish_reason"] == b["finish_reason"]


def first_divergence(a, b):
    sa, sb, unit = sequences(a, b)
    i = next((k for k, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
    return i, unit


def margin(o, i):
    """Top-1 minus top-2 log-probability at generated position i (same prefix on both sides)."""
    top = o.get("top_logprobs")
    if not top or i >= len(top) or len(top[i]) < 2:
        return None
    return top[i][0][1] - top[i][1][1]


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def report(args):
    runs = load_runs(args.runs)
    if not runs:
        sys.exit(f"no runs under {args.runs} (expected <device>/r<k>/outputs.jsonl)")
    cases = {c["prompt_id"]: c for c in read_jsonl(CASES)}
    devices = sorted(runs, key=lambda d: (DEVICE_ORDER.index(d) if d in DEVICE_ORDER else 99, d))
    # Only cases present in every run are compared (supports --limit smoke runs).
    ids = [p for p in cases if all(p in r for dev in runs.values() for r in dev.values())]
    workloads = [w for w in ("code", "math") if any(cases[p]["workload"] == w for p in ids)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scored = all("correct" in r[p] for dev in runs.values() for r in dev.values() for p in ids)

    accuracy = []
    for dev in devices:
        for rep, rows in sorted(runs[dev].items()):
            for w in workloads:
                sel = [rows[p] for p in ids if cases[p]["workload"] == w]
                k = sum(o.get("correct", False) for o in sel) if scored else None
                lo, hi = wilson(k, len(sel)) if scored else (None, None)
                accuracy.append({"device": dev, "repeat": rep, "workload": w, "n": len(sel), "correct": k,
                                 "accuracy_pct": round(100 * k / len(sel), 1) if scored and sel else None,
                                 "ci95_low": lo, "ci95_high": hi})

    # Within-device: is every repeat identical to repeat 1? A case is "stable"
    # on a device only if all its repeats agree.
    repeat_rows, stable = [], {}
    for dev in devices:
        reps = sorted(runs[dev])
        stable[dev] = {p: all(same(runs[dev][reps[0]][p], runs[dev][r][p]) for r in reps) for p in ids}
        for w in workloads:
            sel = [p for p in ids if cases[p]["workload"] == w]
            for r1, r2 in itertools.combinations(reps, 2):
                ident = sum(same(runs[dev][r1][p], runs[dev][r2][p]) for p in sel)
                repeat_rows.append({"device": dev, "repeat_a": r1, "repeat_b": r2, "workload": w, "n": len(sel),
                                    "identical_pct": round(100 * ident / len(sel), 1) if sel else None})

    # Cross-device: repeat 1 on each side; the stable flag says whether repeats confirm it.
    cross_rows, div_rows, week2 = [], [], []
    for da, db in itertools.combinations(devices, 2):
        A, B = runs[da][min(runs[da])], runs[db][min(runs[db])]
        for w in workloads:
            sel = [p for p in ids if cases[p]["workload"] == w]
            ident = sum(same(A[p], B[p]) for p in sel)
            both_stable = [p for p in sel if stable[da][p] and stable[db][p]]
            row = {"device_a": da, "device_b": db, "workload": w, "n": len(sel),
                   "identical_pct": round(100 * ident / len(sel), 1) if sel else None,
                   "output_drift_pct": round(100 * (len(sel) - ident) / len(sel), 1) if sel else None,
                   "stable_on_both": len(both_stable),
                   "stable_and_different": sum(not same(A[p], B[p]) for p in both_stable)}
            if scored:
                ca = [A[p]["correct"] for p in sel]
                cb = [B[p]["correct"] for p in sel]
                bc = sum(x and y for x, y in zip(ca, cb))
                bw = sum(not x and not y for x, y in zip(ca, cb))
                oa = sum(x and not y for x, y in zip(ca, cb))
                ob = sum(y and not x for x, y in zip(ca, cb))
                acc_a, acc_b = 100 * sum(ca) / len(sel), 100 * sum(cb) / len(sel)
                row.update(accuracy_a_pct=round(acc_a, 1), accuracy_b_pct=round(acc_b, 1),
                           accuracy_drift_pp=round(acc_b - acc_a, 1),
                           both_correct=bc, both_wrong=bw, only_a_correct=oa, only_b_correct=ob,
                           flip_pct=round(100 * (oa + ob) / len(sel), 1) if sel else None,
                           mcnemar_p=round(mcnemar(oa, ob), 4))
            cross_rows.append(row)
            for p in sel:
                if same(A[p], B[p]):
                    continue
                i, unit = first_divergence(A[p], B[p])
                ta = A[p]["token_ids"][i] if unit == "token" and i < len(A[p]["token_ids"]) else None
                tb = B[p]["token_ids"][i] if unit == "token" and i < len(B[p]["token_ids"]) else None
                d = {"device_a": da, "device_b": db, "workload": w, "prompt_id": p, "position": i, "unit": unit,
                     "token_a": ta, "token_b": tb,
                     "margin_a": margin(A[p], i) if unit == "token" else None,
                     "margin_b": margin(B[p], i) if unit == "token" else None,
                     "stable_on_both": stable[da][p] and stable[db][p],
                     "correct_a": A[p].get("correct"), "correct_b": B[p].get("correct")}
                div_rows.append(d)
                if d["stable_on_both"] and unit == "token" and ta is not None and tb is not None:
                    week2.append({"device_a": da, "device_b": db, "prompt_id": p, "workload": w,
                                  "flip": scored and A[p]["correct"] != B[p]["correct"],
                                  "input_ids": cases[p]["input_ids"],
                                  "shared_generated_ids": A[p]["token_ids"][:i],
                                  "next_token_a": ta, "next_token_b": tb})

    write_csv(out / "accuracy.csv", accuracy)
    write_csv(out / "repeatability.csv", repeat_rows)
    write_csv(out / "cross_device.csv", cross_rows)
    write_csv(out / "divergences.csv", div_rows)
    (out / "week2-cases.json").write_text(json.dumps(week2))
    write_summary(out, devices, runs, ids, accuracy, repeat_rows, cross_rows, div_rows, week2, scored)
    print((out / "summary.md").read_text())


def write_summary(out, devices, runs, ids, accuracy, repeat_rows, cross_rows, div_rows, week2, scored):
    L = [f"# Week 1 report: {MODEL}", "",
         f"Cases compared: {len(ids)}. Devices: " + ", ".join(f"{d} ({len(runs[d])} repeats)" for d in devices) + ".",
         "" if scored else "\n**Not scored yet** - run `week1.py score` for accuracy and flips.\n"]
    if scored:
        L += ["## Accuracy", "", "Repeat 1; 95% Wilson interval in brackets.", "",
              "| device | workload | correct | accuracy |", "|---|---|---:|---:|"]
        for a in accuracy:
            if a["repeat"] == min(runs[a["device"]]):
                L.append(f"| {a['device']} | {a['workload']} | {a['correct']}/{a['n']} | "
                         f"**{a['accuracy_pct']}%** [{a['ci95_low']}-{a['ci95_high']}] |")
    L += ["", "## Drift (device B relative to device A)", "",
          "*Output drift*: share of answers whose generated text/tokens differ. "
          "*Accuracy drift*: accuracy B minus accuracy A, in percentage points. "
          "*Flip rate*: share of questions whose correctness changed (either direction).", "",
          "| A vs B | workload | output drift | accuracy A | accuracy B | accuracy drift | flip rate |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in cross_rows:
        L.append(f"| {r['device_a']} vs {r['device_b']} | {r['workload']} | {r['output_drift_pct']}% | "
                 + (f"{r['accuracy_a_pct']}% | {r['accuracy_b_pct']}% | {r['accuracy_drift_pp']:+} pp | "
                    f"{r['flip_pct']}% |" if scored else "- | - | - | - |"))
    L += ["", "## 1. Gate: is each device repeatable?", "",
         "Cross-device differences only mean something if repeats on one device are identical.", ""]
    for dev in devices:
        rows = [r for r in repeat_rows if r["device"] == dev]
        worst = min((r["identical_pct"] for r in rows), default=None)
        verdict = "no repeat pairs" if worst is None else ("PASS" if worst == 100 else "FAIL - inspect before comparing devices")
        L.append(f"- **{dev}**: lowest repeat-vs-repeat identity {worst}% -> {verdict}")
    L += ["", "## 2. Cross-device (repeat 1 on each side)", "",
          "| pair | workload | identical | stable & different | both ok | both wrong | only A ok | only B ok | flip | McNemar p |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cross_rows:
        L.append(f"| {r['device_a']} vs {r['device_b']} | {r['workload']} | {r['identical_pct']}% | "
                 f"{r['stable_and_different']}/{r['stable_on_both']} | {r.get('both_correct', '-')} | "
                 f"{r.get('both_wrong', '-')} | {r.get('only_a_correct', '-')} | {r.get('only_b_correct', '-')} | "
                 f"{r.get('flip_pct', '-')}{'%' if scored else ''} | {r.get('mcnemar_p', '-')} |")
    L += ["", "McNemar tests whether one side is systematically more accurate; a large p with many flips "
          "means answers change in both directions.", "",
          "## 3. How close were divergences to a tie?", ""]
    for da, db in itertools.combinations(devices, 2):
        rows = [r for r in div_rows if r["device_a"] == da and r["device_b"] == db]
        m = [r["margin_a"] for r in rows] + [r["margin_b"] for r in rows]
        if any(x is not None for x in m):
            allm = [margin(o, i) for o in runs[da][min(runs[da])].values()
                    for i in range(len(o.get("top_logprobs") or []))]
            L.append(f"- {da} vs {db}: median top-1/top-2 log-prob margin at the first divergent token "
                     f"**{median(m):.4f}** vs **{median(allm):.4f}** over all generated tokens on {da}.")
        else:
            L.append(f"- {da} vs {db}: no log-probabilities available (e.g. on-device sampling).")
    L += ["", "## Files", "",
          "- `accuracy.csv`: accuracy per device/repeat/workload with Wilson 95% CI",
          "- `repeatability.csv`: identity between repeats on each device",
          "- `cross_device.csv`: identity, four outcome counts, flip rate, McNemar p",
          "- `divergences.csv`: first divergent position, tokens and margins per differing case",
          f"- `week2-cases.json`: {len(week2)} stable divergent cases with the shared prefix (input for Week 2)"]
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--device", required=True)
    g.add_argument("--repeat", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--url", default="http://127.0.0.1:8001")
    g.add_argument("--limit", type=int, help="smoke test: first N cases per workload")
    g.add_argument("--logprobs", type=int, default=5, help="top-k log-probs per token; 0 disables")
    g.add_argument("--no-token-ids", action="store_true", help="for servers that cannot return token IDs")
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+", help="run folders containing outputs.jsonl")
    r = sub.add_parser("report")
    r.add_argument("--runs", required=True, help="folder with <device>/r<k>/ subfolders")
    r.add_argument("--out", required=True)
    args = p.parse_args()
    {"generate": generate, "score": score, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
