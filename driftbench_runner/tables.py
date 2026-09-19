"""Portable per-prompt tables, collected matrices, and readable paired results."""
import csv
import html
import json
from pathlib import Path
import re

from .common import WORKLOADS, digest, now, write_json
from .evaluation import load_run
from .logging import event

BENCHMARKS = {"code": "HumanEval", "math": "GSM8K", "safety": "AdvBench",
              "chat": "LMSYS-Chat-1M", "long_context": "LongBench-Qasper"}
ROW_FIELDS = ["setting_id", "gpu", "gpu_vendor", "model", "model_revision", "framework", "framework_version",
              "transport", "workload", "benchmark", "row_number", "row_id", "prompt_id", "result", "correct",
              "score", "label", "severity", "evaluation_status", "evaluation_method", "evaluator_version",
              "reason", "prompt", "output_text", "reference", "extracted_answer", "input_tokens", "output_tokens",
              "finish_reason", "latency_seconds", "source_sha256", "request_sha256", "output_sha256", "run_fingerprint"]
ROW_FIELDS += ["threshold", "judge", "execution_environment", "detail", "inference_status"]


def row_order(key):
    workload, prompt_id = key
    parts = tuple((1, int(p)) if p.isdigit() else (0, p) for p in re.split(r"(\d+)", str(prompt_id)))
    return (list(WORKLOADS).index(workload) if workload in WORKLOADS else len(WORKLOADS), workload, parts)


def outcome(evaluation):
    status = evaluation.get("status", "pending")
    if status != "scored":
        return status
    if evaluation.get("label") in ("safe", "unsafe"):
        return evaluation["label"]
    if type(evaluation.get("correct")) is not bool:
        raise ValueError("A scored row must have an explicit boolean correctness label")
    return "correct" if evaluation["correct"] else "incorrect"


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items() if k in fields})
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def setting_rows(run_dir):
    # Imports are local to keep comparison.py able to use these serializers.
    from .comparison import verified_evaluations
    manifest, outputs = load_run(run_dir)
    scores = verified_evaluations(run_dir, manifest, outputs)
    config = manifest["config"]
    setting_id = config.get("setup_id", Path(run_dir).name)
    numbers = {}
    rows = []
    for key in sorted(outputs, key=row_order):
        raw = outputs[key]
        score = scores.get(key, {"status": "pending", "reason": "Evaluation has not been saved for this response"})
        if raw.get("status", "ok") != "ok":
            score = {"status": "error", "reason": "Inference failed"}
        if score.get("source_sha256", raw["source_sha256"]) != raw["source_sha256"]:
            raise ValueError(f"Stale evaluation source: {key}")
        numbers[key[0]] = numbers.get(key[0], 0) + 1
        row = {"setting_id": setting_id, "gpu": manifest["environment"].get("gpu_name", config.get("hardware", {}).get("device")),
               "gpu_vendor": config.get("hardware", {}).get("vendor"), "model": config["model"],
               "model_revision": config["revision"], "framework": config["backend"],
               "framework_version": manifest["environment"].get("packages", {}).get(config["backend"]),
               "transport": config.get("transport", "offline"), "workload": key[0],
               "benchmark": BENCHMARKS.get(key[0], key[0]), "row_number": numbers[key[0]],
               "row_id": f"{key[0]}:{key[1]}", "prompt_id": key[1], "result": outcome(score),
               "correct": score.get("correct") if score.get("status") == "scored" else None,
               "score": score.get("score"), "label": score.get("label"), "severity": score.get("severity"),
               "evaluation_status": score.get("status", "pending"), "evaluation_method": score.get("method"),
               "evaluator_version": score.get("evaluator_version"), "reason": score.get("reason"),
               "reference": score.get("reference"), "extracted_answer": score.get("extracted_answer"),
               "output_sha256": digest(raw["output_text"]), "run_fingerprint": manifest["run_fingerprint"]}
        row.update({k: raw.get(k) for k in ("prompt", "output_text", "input_tokens", "output_tokens", "finish_reason",
                                           "latency_seconds", "source_sha256", "request_sha256")})
        row.update({k: score.get(k) for k in ("threshold", "judge", "execution_environment", "detail")})
        row["inference_status"] = raw.get("status", "ok")
        rows.append(row)
    return manifest, rows


def export_run(run_dir, output_dir=None):
    out = Path(output_dir) if output_dir else Path(run_dir) / "tables"
    manifest, rows = setting_rows(run_dir)
    write_csv(out / "rows.csv", rows, ROW_FIELDS)
    write_jsonl(out / "rows.jsonl", rows)
    for workload in manifest["sources"]:
        write_csv(out / f"{workload}.csv", [r for r in rows if r["workload"] == workload], ROW_FIELDS)
    for workload in set(WORKLOADS) - set(manifest["sources"]):
        (out / f"{workload}.csv").unlink(missing_ok=True)
    metadata = {"schema_version": 1, "setting_id": manifest["config"].get("setup_id", Path(run_dir).name),
                "run_fingerprint": manifest["run_fingerprint"], "inference_status": manifest["status"],
                "expected_records": manifest["expected_records"], "exported_records": len(rows),
                "join_key": ["workload", "prompt_id"], "config": manifest["config"],
                "environment": manifest["environment"], "sources": manifest["sources"],
                "meaning": "Safety uses safe/unsafe. Chat is pairwise_only. Pending/error never means incorrect."}
    write_json(out / "setting.json", metadata)
    event("Response tables saved", stage="export", setting=metadata["setting_id"], completed=len(rows), output=str(out))
    return rows


def collect_runs(run_dirs, output_dir, aliases=None):
    out = Path(output_dir)
    aliases = aliases or {}
    settings, all_rows, maps = [], [], {}
    sources = {}
    for directory in run_dirs:
        manifest, rows = setting_rows(directory)
        sid = aliases.get(str(directory), manifest["config"].get("setup_id", Path(directory).name))
        if sid in maps:
            raise ValueError(f"Duplicate setting ID {sid!r}; use --aliases or unique setup_id values")
        for row in rows:
            row["setting_id"] = sid
            key = (row["workload"], row["prompt_id"])
            if key in sources and sources[key] != row["source_sha256"]:
                raise ValueError(f"Source prompt/reference differs across settings: {key}")
            sources[key] = row["source_sha256"]
        maps[sid] = {(r["workload"], r["prompt_id"]): r for r in rows}
        all_rows.extend(rows)
        settings.append({"setting_id": sid, "run_dir": str(Path(directory).resolve()),
                         "run_fingerprint": manifest["run_fingerprint"], "inference_status": manifest["status"],
                         "expected_records": manifest["expected_records"], "exported_records": len(rows),
                         "config": manifest["config"], "environment": manifest["environment"], "sources": manifest["sources"]})
    if not settings:
        raise ValueError("Select at least one saved run")
    write_csv(out / "rows.csv", all_rows, ROW_FIELDS)
    write_jsonl(out / "rows.jsonl", all_rows)
    write_json(out / "settings.json", {"schema_version": 1, "created_at": now(), "settings": settings,
                                      "join_key": ["workload", "prompt_id"],
                                      "missing": "No output for this ID in that run; it may be unselected or unfinished.",
                                      "warning": "Matrices gather observations. Use compare for evaluator/control compatibility checks and drift rates."})
    summaries = []
    for item in settings:
        sid = item["setting_id"]
        for workload, source in item["sources"].items():
            group = [r for r in maps[sid].values() if r["workload"] == workload]
            summaries.append({"setting_id": sid, "workload": workload, "benchmark": BENCHMARKS[workload],
                              "expected": source["selected"], "rows": len(group), "missing": source["selected"] - len(group),
                              "inference_status": item["inference_status"], **{label: sum(r["result"] == label for r in group)
                               for label in ("correct", "incorrect", "safe", "unsafe", "pairwise_only", "pending", "error")}})
    write_csv(out / "settings.csv", summaries, ["setting_id", "workload", "benchmark", "expected", "rows", "missing", "inference_status", "correct", "incorrect",
                                               "safe", "unsafe", "pairwise_only", "pending", "error"])
    fields = ["row_id", "workload", "benchmark", "prompt_id", "prompt"]
    values = ("result", "correct", "score", "output_text", "evaluation_status")
    fields += [f"{sid}.{field}" for sid in maps for field in values]
    matrix = []
    for key in sorted(sources, key=row_order):
        sample = next(m[key] for m in maps.values() if key in m)
        row = {k: sample[k] for k in ("row_id", "workload", "benchmark", "prompt_id", "prompt")}
        for sid, mapping in maps.items():
            value = mapping.get(key, {"result": "missing", "evaluation_status": "missing"})
            row.update({f"{sid}.{k}": value.get(k) for k in values})
        matrix.append(row)
    write_csv(out / "matrix.csv", matrix, fields)
    for workload in dict.fromkeys(r["workload"] for r in matrix):
        write_csv(out / "workloads" / f"{workload}.csv", [r for r in matrix if r["workload"] == workload], fields)
    for workload in set(WORKLOADS) - {r["workload"] for r in matrix}:
        (out / "workloads" / f"{workload}.csv").unlink(missing_ok=True)
    event("Settings gathered", stage="collection", settings=len(settings), responses=len(all_rows), prompt_rows=len(matrix), output=str(out))
    return {"settings": settings, "rows": len(all_rows), "prompt_rows": len(matrix)}


def write_paired_tables(report, baseline_dir, candidate_dir, output_dir):
    _, left_rows = setting_rows(baseline_dir)
    _, right_rows = setting_rows(candidate_dir)
    left = {(r["workload"], r["prompt_id"]): r for r in left_rows}
    right = {(r["workload"], r["prompt_id"]): r for r in right_rows}
    rows = []
    for pair in sorted(report["records"], key=lambda r: row_order((r["workload"], r["prompt_id"]))):
        key = (pair["workload"], pair["prompt_id"])
        a, b = left[key], right[key]
        row = {"row_id": a["row_id"], "workload": key[0], "benchmark": a["benchmark"], "prompt_id": key[1],
               "prompt": a["prompt"], **pair}
        for prefix, source in (("baseline", a), ("candidate", b)):
            for field in ("setting_id", "result", "correct", "score", "evaluation_status", "output_text",
                          "model", "model_revision", "gpu", "framework", "framework_version", "reference", "finish_reason"):
                row[f"{prefix}_{field}"] = source[field]
        rows.append(row)
    fields = ["row_id", "workload", "benchmark", "prompt_id", "baseline_setting_id", "candidate_setting_id",
              "baseline_result", "candidate_result", "label_flip", "text_changed", "token_sequence_changed",
              "baseline_correct", "candidate_correct", "baseline_score", "candidate_score", "semantic_shift",
              "substantial_semantic_drift", "prompt", "baseline_output_text", "candidate_output_text"]
    fields += sorted(set().union(*(r.keys() for r in rows)) - set(fields))
    out = Path(output_dir)
    write_csv(out / "rows.csv", rows, fields)
    write_jsonl(out / "rows.jsonl", rows)
    for workload in report["workloads"]:
        write_csv(out / "workloads" / f"{workload}.csv", [r for r in rows if r["workload"] == workload], fields)
    for workload in set(WORKLOADS) - set(report["workloads"]):
        (out / "workloads" / f"{workload}.csv").unlink(missing_ok=True)
    write_browser(out / "rows.html", rows, report)


def write_browser(path, rows, report):
    # Escape embedded data and use textContent throughout. Model output is data.
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    note = html.escape("Changed factors: " + (", ".join(report["changed_factors"]) or "none") +
                       ". Changed controls: " + (", ".join(report["confounds"]) or "none") +
                       ". " + report.get("evaluation_note", ""))
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DriftBench row comparison</title>
<style>body{font:16px system-ui;margin:2rem;max-width:1500px}select,input{font:inherit;padding:.4rem;margin:.3rem}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border:1px solid #ddd;padding:.6rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}pre{white-space:pre-wrap;word-break:break-word;max-height:28rem;overflow:auto;font:14px monospace}.change{background:#fff1c2}details{padding:.6rem;border-bottom:1px solid #ddd}summary{cursor:pointer}small{color:#555}</style>
<h1>DriftBench row comparison</h1><p>__NOTE__</p>
<label>Workload <select id="workload"><option value="">All</option></select></label>
<label>Show <select id="filter"><option value="all">All rows</option><option value="flips">Label flips</option><option value="semantic">Substantial chat drift</option><option value="text">Changed responses</option><option value="unscored">Unscored labels</option></select></label>
<label>Search <input id="search" placeholder="Prompt ID or response text"></label><p id="count"></p><main id="rows"></main>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), w=document.getElementById('workload'), f=document.getElementById('filter'), q=document.getElementById('search');
for(const name of new Set(data.map(r=>r.workload))){const o=document.createElement('option');o.value=name;o.textContent=name;w.appendChild(o)}
function render(){
  const all=data.filter(r=>(!w.value||r.workload===w.value)&&
    (f.value==='all'||f.value==='flips'&&r.label_flip===true||f.value==='text'&&r.text_changed||
     f.value==='semantic'&&r.substantial_semantic_drift===true||f.value==='unscored'&&r.label_flip===null)&&
    JSON.stringify(r).toLowerCase().includes(q.value.toLowerCase()));
  document.getElementById('count').textContent=all.length+' matching rows (first 100 shown; filter by workload or prompt ID for more).';
  const root=document.getElementById('rows');root.replaceChildren();
  for(const r of all.slice(0,100)){
    const d=document.createElement('details'),s=document.createElement('summary');
    s.textContent=r.row_id+' — '+r.baseline_result+' → '+r.candidate_result+
      (r.label_flip?' | LABEL FLIP':'')+(r.text_changed?' | text changed':'')+
      (r.semantic_shift!=null?' | cosine shift: '+r.semantic_shift.toFixed(4):'')+
      (r.substantial_semantic_drift?' | SUBSTANTIAL CHAT DRIFT':'');
    if(r.label_flip||r.substantial_semantic_drift)d.className='change';d.appendChild(s);
    const prompt=document.createElement('pre');prompt.textContent=r.prompt;d.appendChild(prompt);
    if(r.embedding_input_truncated){const note=document.createElement('p');
      note.textContent='Embedding evaluator truncated at least one response; full responses are shown below.';d.appendChild(note)}
    const table=document.createElement('table'),head=document.createElement('tr'),body=document.createElement('tr');
    for(const side of ['baseline','candidate']){
      const th=document.createElement('th');th.textContent=r[side+'_setting_id']+' | '+r[side+'_result']+
        ' | score: '+(r[side+'_score']??'not scored');head.appendChild(th);
      const td=document.createElement('td'),pre=document.createElement('pre');
      pre.textContent=r[side+'_output_text'];td.appendChild(pre);body.appendChild(td)
    }
    table.append(head,body);d.appendChild(table);root.appendChild(d)
  }
}
w.onchange=f.onchange=q.oninput=render;render();
</script>'''
    Path(path).write_text(page.replace("__NOTE__", note).replace("__DATA__", payload), encoding="utf-8")
