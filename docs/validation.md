# Validation on this machine

The complete baseline ran on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), using Qwen3.5-0.8B BF16 and vLLM 0.17.1. All **2,284/2,284 responses** are saved in `results/rtx3060-qwen35-vllm/`. All **1,284 objective responses** have scores; the remaining 1,000 chat responses require a second run for semantic-drift evaluation. No inference records or objective scores are missing or pending. See [the results report](results.md).

Completed checks:

- All published inputs loaded with unique workload/prompt IDs and SHA-256 dataset fingerprints.
- Full-input tokenizer preflight: maximum 21,688 input tokens; 22,200 with the response budget. No input truncation was required.
- All 239 fixed batches completed: 11 code, 32 math, 33 safety, 63 chat and 100 long-context batches. Batch membership, token counts and the run fingerprint passed the final audit.
- Qwen3Guard-Gen-0.6B classified all 520 generated safety responses on the GPU. Raw judgments, ternary labels, revision, hardware/library provenance and response hashes are retained.
- All **164/164 canonical HumanEval solutions** passed in isolated workers with evaluator v3 (`results/humaneval-harness-validation-v3.json`).
- **29 tests passed**, covering score extraction, Wilson intervals, source alignment, stale-evaluation rejection, duplicate IDs, resume controls, fixed batch membership, filesystem isolation, judge-hardware consistency, supplied Python helpers, Markdown extraction, deterministic worker seeds, code-evaluation host consistency and Unicode JSONL records.
- `scripts/audit_baseline.py` passed against the completed run: exact source IDs and hashes, original prompts, request fingerprints, token limits, batch assignments, evaluation coverage, judge consistency and summary totals. Its result and data-file checksums are saved in `results/rtx3060-qwen35-vllm/audit.json`.

The chat audit found valid Unicode line separators inside two source prompts. The JSONL reader now uses physical record boundaries instead of Python's broader `str.splitlines()`. All 1,000 chat records were intact; no inference output was changed. Source hashes before and after that reader fix are preserved in the provenance files.

The full-run startup initially failed to allocate cache memory while another GPU worker appeared. That attempt produced no inference records; its log remains in `results/full-run.log`. The unchanged profile subsequently initialized with 1.96 GiB of KV cache and completed successfully. Another inference job shared the GPU during much of the run, so batch durations are not measurements of exclusive GPU performance. The successful pipeline log is `results/full-run-retry.log`.

The earlier smoke run in `results/rtx3060-smoke/` contains 10 real responses, two per workload. Its CPU safety labels with hardware provenance are in `safety-labels-with-environment.jsonl`. The embedding self-comparison in `results/smoke-self-check/` validates report generation only; comparing a file to itself is not a repeatability experiment or evidence about hardware drift. Encoder truncation is recorded separately from inference input truncation.

The full HumanEval result is 37/164 with evaluator v3. A preliminary v1 count of 36/164 was superseded after fixing prompt-helper preservation and Markdown extraction. The saved model outputs were unchanged.

No A100, AMD, Tenstorrent, SGLang or TensorRT-LLM run has been performed here. The matching A100 configuration and paired comparison workflow are prepared. Re-evaluate both hardware runs on the same code-test and safety-judge host before comparing labels.
