import csv
from pathlib import Path

from .common import ROOT, digest, keyed, local_environment, read_json, write_json
from .evaluation import load_run, wilson
from .logging import activity, event


def verified_evaluations(directory, manifest, outputs):
    path = Path(directory) / "evaluation.json"
    if not path.exists():
        return {}
    evaluation = read_json(path)
    if evaluation["summary"]["run_fingerprint"] != manifest["run_fingerprint"]:
        raise ValueError("Evaluation belongs to a different run")
    result = keyed(evaluation["records"])
    for key, score in result.items():
        if key not in outputs or score["output_sha256"] != digest(outputs[key]["output_text"]):
            raise ValueError(f"Stale evaluation: {key}")
        if score.get("source_sha256", outputs[key]["source_sha256"]) != outputs[key]["source_sha256"]:
            raise ValueError(f"Stale evaluation source: {key}")
    return result


def compare_runs(baseline_dir, candidate_dir, output_dir, semantic=False, allow_confounded=False):
    event("Checking paired responses and evaluation compatibility", stage="comparison", baseline=str(baseline_dir), candidate=str(candidate_dir))
    baseline_meta, baseline = load_run(baseline_dir)
    candidate_meta, candidate = load_run(candidate_dir)
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("Comparison requires nonempty, identical prompt-ID sets; incomplete overlap is not a zero drift result")
    for metadata, outputs in ((baseline_meta, baseline), (candidate_meta, candidate)):
        if metadata["status"] != "complete" or len(outputs) != metadata["expected_records"]:
            raise ValueError("Finish inference before comparison")
    confounds = []
    if baseline_meta.get("scheduling", "offline_batches") != candidate_meta.get("scheduling", "offline_batches"):
        confounds.append("request_scheduling")
    def launch_controls(metadata):
        return {k: v for k, v in metadata["config"].get("launch", {}).items()
                if k not in ("log_path", "startup_timeout_seconds")}
    if launch_controls(baseline_meta) != launch_controls(candidate_meta):
        confounds.append("launch_settings")
    for field in ("model", "revision", "generation", "prompt_format", "chat_template_kwargs", "seed", "batch_size", "batch_size_by_workload"):
        if baseline_meta["config"].get(field) != candidate_meta["config"].get(field):
            confounds.append(field)
    changed_factors = []
    for factor, a, b in (
        ("hardware", baseline_meta["environment"].get("gpu_name"), candidate_meta["environment"].get("gpu_name")),
        ("framework", baseline_meta["config"]["backend"], candidate_meta["config"]["backend"]),
        ("precision", baseline_meta["config"]["engine"].get("dtype"), candidate_meta["config"]["engine"].get("dtype")),
    ):
        if a != b:
            changed_factors.append(factor)
    engine_diff = {k: [baseline_meta["config"]["engine"].get(k), candidate_meta["config"]["engine"].get(k)]
                   for k in set(baseline_meta["config"]["engine"]) | set(candidate_meta["config"]["engine"])
                   if baseline_meta["config"]["engine"].get(k) != candidate_meta["config"]["engine"].get(k)}
    if set(engine_diff) - {"dtype"}:
        confounds.append("engine_parameters")
    for field in ("packages", "cuda_runtime", "rocm_runtime", "package_sources", "source_revisions"):
        default = {} if field in ("package_sources", "source_revisions") else None
        if baseline_meta["environment"].get(field, default) != candidate_meta["environment"].get(field, default):
            confounds.append(f"environment_{field}")
    if len(changed_factors) > 1:
        confounds.append("multiple_setup_factors")
    for key in baseline:
        if baseline[key]["source_sha256"] != candidate[key]["source_sha256"]:
            raise ValueError(f"Source prompt/reference differs: {key}")
        if baseline[key]["request_sha256"] != candidate[key]["request_sha256"]:
            confounds.append("rendered_prompts_or_tokenization")
    confounds = sorted(set(confounds))
    if confounds and not allow_confounded:
        raise ValueError(f"Changed experimental controls: {confounds}. Use --allow-confounded to label this explicitly")
    base_eval = verified_evaluations(baseline_dir, baseline_meta, baseline)
    cand_eval = verified_evaluations(candidate_dir, candidate_meta, candidate)
    pairs = []
    for key in baseline:
        a, b = baseline[key], candidate[key]
        row = {"workload": key[0], "prompt_id": key[1], "source_sha256": a["source_sha256"],
               "text_changed": a["output_text"] != b["output_text"],
               "token_sequence_changed": a["output_token_ids"] != b["output_token_ids"] if
               baseline_meta["config"]["model"] == candidate_meta["config"]["model"] and
               baseline_meta["config"]["revision"] == candidate_meta["config"]["revision"] and
               a.get("output_token_ids") is not None and b.get("output_token_ids") is not None else None,
               "baseline_output_sha256": digest(a["output_text"]), "candidate_output_sha256": digest(b["output_text"]),
               "label_flip": None, "baseline_label": None, "candidate_label": None,
               "baseline_length_limited": a["finish_reason"] == "length",
               "candidate_length_limited": b["finish_reason"] == "length",
               "output_length_ratio_chars": len(b["output_text"]) / len(a["output_text"]) if a["output_text"] else None}
        ea, eb = base_eval.get(key, {}), cand_eval.get(key, {})
        if ea.get("status") == eb.get("status") == "scored":
            for field in ("method", "evaluator_version", "threshold", "judge", "execution_environment"):
                if ea.get(field) != eb.get(field):
                    raise ValueError(f"Evaluators differ for {key}: {field}")
            la, lb = ea.get("label", ea["correct"]), eb.get("label", eb["correct"])
            if not row["text_changed"] and la != lb:
                raise ValueError(f"Identical response received different labels for {key}; re-evaluate on a common host")
            row.update(baseline_label=la, candidate_label=lb, label_flip=la != lb,
                       baseline_score=ea["score"], candidate_score=eb["score"],
                       evaluation_method=ea.get("method"), evaluator_version=ea.get("evaluator_version"))
            if "judge" in ea:
                row["safety_judge"] = ea["judge"]
            if "execution_environment" in ea:
                row["code_execution_environment"] = ea["execution_environment"]
        pairs.append(row)
    semantic_meta = None
    if semantic:
        with activity("Loading chat embedding evaluator", stage="model_load", device="cpu"):
            local_environment()
            from sentence_transformers import SentenceTransformer
            pinned = read_json(ROOT / "models.lock.json")["chat_embedding"]
            model_id, revision = pinned["model"], pinned["revision"]
            model = SentenceTransformer(model_id, revision=revision, device="cpu")
        chat = [r for r in pairs if r["workload"] == "chat"]
        if chat:
            a_texts = [baseline[("chat", r["prompt_id"])]["output_text"] for r in chat]
            b_texts = [candidate[("chat", r["prompt_id"])]["output_text"] for r in chat]
            with activity("Embedding baseline chat responses", stage="comparison", workload="chat", total=len(chat), device="cpu") as progress:
                emb_a = model.encode(a_texts, normalize_embeddings=True, show_progress_bar=False)
                progress.update(len(chat))
            with activity("Embedding candidate chat responses", stage="comparison", workload="chat", total=len(chat), device="cpu") as progress:
                emb_b = model.encode(b_texts, normalize_embeddings=True, show_progress_bar=False)
                progress.update(len(chat))
            for row, va, vb, ta, tb in zip(chat, emb_a, emb_b, a_texts, b_texts):
                similarity = min(1.0, max(-1.0, float(va @ vb)))
                row.update(cosine_similarity=similarity, semantic_shift=1.0 - similarity,
                           substantial_semantic_drift=(1.0 - similarity) > 0.3,
                           embedding_input_truncated=any(len(model.tokenizer.encode(t, verbose=False)) > model.max_seq_length for t in (ta, tb)))
        semantic_meta = {"model": model_id, "revision": revision, "max_seq_length": model.max_seq_length,
                         "threshold": "1 - cosine_similarity > 0.3", "device": "cpu"}
    summary = {}
    for workload in baseline_meta["sources"]:
        rows = [r for r in pairs if r["workload"] == workload]
        labels = [r for r in rows if r["label_flip"] is not None]
        flips = sum(r["label_flip"] for r in labels)
        semantic_rows = [r for r in rows if "semantic_shift" in r]
        semantic_flips = sum(r["substantial_semantic_drift"] for r in semantic_rows)
        token_pairs = [r for r in rows if r["token_sequence_changed"] is not None]
        summary[workload] = {"paired": len(rows), "text_changes": sum(r["text_changed"] for r in rows),
                             "text_change_rate": sum(r["text_changed"] for r in rows) / len(rows),
                             "token_comparable_pairs": len(token_pairs),
                             "token_change_rate": sum(r["token_sequence_changed"] for r in token_pairs) / len(token_pairs) if token_pairs else None,
                             "scored_pairs": len(labels), "unscored_pairs": len(rows) - len(labels),
                             "label_flips": flips if labels else None,
                             "flip_rate": flips / len(labels) if labels else None,
                             "flip_rate_wilson95": wilson(flips, len(labels)),
                             "correct_to_incorrect": sum(r["baseline_label"] is True and r["candidate_label"] is False for r in labels),
                             "incorrect_to_correct": sum(r["baseline_label"] is False and r["candidate_label"] is True for r in labels),
                             "safe_to_unsafe": sum(r["baseline_label"] == "safe" and r["candidate_label"] == "unsafe" for r in labels),
                             "unsafe_to_safe": sum(r["baseline_label"] == "unsafe" and r["candidate_label"] == "safe" for r in labels),
                             "semantic_pairs": len(semantic_rows),
                             "embedding_truncated_pairs": sum(r["embedding_input_truncated"] for r in semantic_rows),
                             "mean_semantic_shift": sum(r["semantic_shift"] for r in semantic_rows) / len(semantic_rows) if semantic_rows else None,
                             "substantial_semantic_drift_rate": semantic_flips / len(semantic_rows) if semantic_rows else None,
                             "semantic_drift_wilson95": wilson(semantic_flips, len(semantic_rows))}
    report = {"schema_version": 1, "baseline": str(baseline_dir), "candidate": str(candidate_dir),
              "comparison_kind": "self_check" if Path(baseline_dir).resolve() == Path(candidate_dir).resolve() else "paired_runs",
              "baseline_fingerprint": baseline_meta["run_fingerprint"], "candidate_fingerprint": candidate_meta["run_fingerprint"],
              "changed_factors": changed_factors, "confounds": confounds, "engine_differences": engine_diff,
              "baseline_environment": baseline_meta["environment"], "candidate_environment": candidate_meta["environment"],
              "semantic_evaluator": semantic_meta, "workloads": summary, "records": pairs}
    output_dir = Path(output_dir)
    write_json(output_dir / "comparison.json", report)
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["workload", *next(iter(summary.values())).keys()])
        writer.writeheader()
        writer.writerows({"workload": name, **value} for name, value in summary.items())
    from .tables import write_paired_tables
    write_paired_tables(report, baseline_dir, candidate_dir, output_dir)
    event("Comparison saved", stage="comparison", paired=len(pairs),
          text_changes=sum(r["text_changed"] for r in pairs),
          scored_pairs=sum(r["label_flip"] is not None for r in pairs),
          label_flips=sum(r["label_flip"] is True for r in pairs), viewer=str(output_dir / "rows.html"))
    return report
