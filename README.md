# DriftBench inference experiments

This project runs fresh inference on the five published DriftBench workloads and saves each response for later comparisons across machines. The first profile targets an NVIDIA RTX 3060 Laptop GPU (6 GB), `Qwen/Qwen3.5-0.8B`, and vLLM. The model's official name has no `-Instruct` suffix.

This is an adaptation of the [MLSys 2026 DriftBench experiment](https://openreview.net/forum?id=Xfzzp6grRP), using the [authors' prompt files](https://github.com/GianluigiVitale/driftbench-ae/tree/main/artifact_evaluation/reviewer-verification-master/datasets). It does not reproduce the paper's large-model numerical results.

The full RTX 3060 baseline is complete: **2,284 responses saved and all 1,284 objective responses evaluated**. The 1,000 chat responses are ready for a future paired comparison. See the [results report](docs/results.md), [validation](docs/validation.md), and [portable bundle](results/driftbench-rtx3060-bundle.tar.gz).

**Multiple serving frameworks:** NVIDIA vLLM, SGLang and TensorRT-LLM now have real workload validation, saved responses and two paired comparison reports. AMD and Tenstorrent profiles are prepared; physical validation awaits access. See the [platform guide](docs/platforms.md), [framework validation](docs/framework-validation.md), and [updated bundle](results/driftbench-multiplatform-bundle.tar.gz).

| Workload | Published inputs used here | Evaluation |
|---|---:|---|
| HumanEval / code | 164 | Execute tests; one greedy completion, pass@1 |
| GSM8K / math | 500 | Extract final number; exact match |
| AdvBench / safety | 520 | Qwen3Guard-Gen-0.6B response classification; LlamaGuard-3-8B available separately |
| LMSYS-Chat-1M / chat | 1,000 | Pairwise embedding cosine shift; requires another run |
| LongBench Qasper / long context | 100 | Token F1; correctness threshold F1 ≥ 0.5 |

The paper reports 973 chat prompts, but publishes 1,000 without a recoverable list of the excluded IDs. We use all 1,000, giving **2,284** inputs, and retain exact IDs and file hashes. See [methodology and deviations](docs/methodology.md).

## Setup

Linux or WSL2 with a working NVIDIA driver is required for this profile. Downloads and Python packages stay in the workspace; the system Python is not changed. Dependencies are pinned in `requirements.lock.txt`. Allow roughly 20 GB for the Python/CUDA environment, caches and models.

```bash
bash scripts/bootstrap.sh
source scripts/env.sh
nvidia-smi
python -m driftbench_runner preflight
```

The checked profile uses native BF16 weights, text-only loading, 32K context, a 512-token generation budget, seed 42, temperature 0, disabled thinking, one sequence at a time and eager execution. Its GPU reservation is 60% to leave room for the laptop display. All published inputs fit without truncation. Available VRAM still depends on other applications.

## Run inference and evaluation

For the complete sequence (inference, small safety judge, then evaluation), run:

```bash
bash scripts/run_full.sh > results/full-run.log 2>&1
# In another terminal:
python scripts/progress.py results/rtx3060-qwen35-vllm
```

The script resumes an existing compatible run and does not overwrite completed inference records. Individual stages are also available:

```bash
# Two examples from each workload to check installation.
python -m driftbench_runner run --limit 2 --output results/rtx3060-smoke

# Full published workload files with batching. No --limit means all 2,284 inputs.
python -m driftbench_runner run --config configs/rtx3060-qwen35-vllm-batch16.json \
  --output results/rtx3060-qwen35-vllm

# Reuse completed records after interruption; requires identical settings.
python -m driftbench_runner run --config configs/rtx3060-qwen35-vllm-batch16.json \
  --output results/rtx3060-qwen35-vllm --resume

# Run the small safety judge after the inference process releases GPU memory.
python -m driftbench_runner judge-safety results/rtx3060-qwen35-vllm \
  --device cuda --output results/rtx3060-qwen35-vllm/safety-labels.jsonl

python -m driftbench_runner evaluate results/rtx3060-qwen35-vllm --code \
  --safety-labels results/rtx3060-qwen35-vllm/safety-labels.jsonl
```

`--workloads code math` selects workloads. `--limit N` selects the first N from each file and is always recorded as a subset. Offline profiles load the model locally. HTTP profiles send requests to the endpoint explicitly configured in `server.base_url`.

The full-run profile batches 16 requests for code, math, safety and chat, with 2,048 prefill tokens per engine iteration. Long-context requests run one at a time. Both profiles preserve greedy decoding and the 512-token response cap. Batch membership is fixed within each workload and preserved on resume. The matching A100 batch-16 profile must be used when comparing to this full run; the single-sequence smoke profile is a different experiment.

HumanEval runs generated code in bubblewrap with separate namespaces, no network, no home/workspace mount, a temporary filesystem, and time/memory limits. If isolation is unavailable, those scores remain explicitly pending. It never falls back to executing generated code directly on the host.

The smaller safety judge is the user's requested adaptation for 6 GB VRAM. Its three severity labels are preserved. For binary drift, `Safe` maps to safe and both `Unsafe` and `Controversial` map to unsafe; the latter is a declared conservative policy, not the paper's LlamaGuard classification. [Qwen3Guard model card](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B).

To use the paper's judge later, obtain approved access to the gated model, authenticate locally, and run on hardware with sufficient memory:

```bash
python -m driftbench_runner judge-safety results/rtx3060-qwen35-vllm \
  --model meta-llama/Llama-Guard-3-8B --device cuda \
  --output results/rtx3060-qwen35-vllm/llamaguard3-labels.jsonl
```

## Compare with an A100 or another setup

Copy this project and its locked dependencies to the other machine. Keep the model revision, prompt files, generation settings, batch size, prompt format and precision fixed. Change `setup_id` in a copy of the configuration; use `--config` to select it. An A100 can use the same conservative memory settings for a controlled comparison.

```bash
# On the other machine, using a copied config with its own setup_id:
python -m driftbench_runner run --config configs/a100-qwen35-vllm-batch16.json \
  --output results/a100-qwen35-vllm
# Transfer the saved inference directory to the common evaluation machine.

# Bring both result directories together, then:
python -m driftbench_runner compare \
  results/rtx3060-qwen35-vllm results/a100-qwen35-vllm \
  --semantic --output results/3060-vs-a100
```

This reports correctness/safety label flips, direction of flips, Wilson 95% confidence intervals, exact output changes, and chat semantic shift. A text change alone is **not** a correctness flip. Comparisons require identical prompt-ID sets and verified scoring provenance. Changed model/decoding controls require `--allow-confounded` and remain labeled in the report. Hardware and framework metadata are included so additional changes can be inspected.

For a hardware comparison, **evaluate both runs on the same evaluation machine**. HumanEval records its Python/CPU environment and resource limits; workers use random and hash seeds of 42. The safety judge's hardware and library versions are also recorded and must match: running the judge on each target GPU could itself introduce classification drift. You can transfer the A100 inference directory to the 3060 machine and run `judge-safety` and `evaluate --code` for both there, or use the small judge with `--device cpu` for both on a common CPU host. Keep alternate judge outputs in separate files.

The runner now has HTTP adapters and launch profiles for vLLM, SGLang and TensorRT-LLM, plus AMD ROCm and Tenstorrent vendor environments. See [platform setup and validation status](docs/platforms.md). AMD and Tenstorrent hardware validation is pending access to those machines. The pretrained paper PRI predictor is not used to invent drift estimates for this new model/GPU combination.

## Results

Each run directory contains:

- `manifest.json`: immutable model revision, dataset hashes, config, software/GPU metadata, run status.
- `{code,math,safety,chat,long_context}.jsonl`: original and rendered prompts, token IDs, output text, finish reasons, input/output counts and measured batch duration.
- `safety-labels.jsonl`: separate, resumable judge outputs with model/revision and response hashes.
- `evaluation.json` and `summary.csv`: per-prompt scores, workload summaries, missing/pending counts and length-limited responses.

Comparison directories contain `comparison.json` (including every paired result) and `comparison.csv`. Fractions are in [0, 1]. The original baseline uses vLLM's offline engine. New HTTP runs preserve per-request latency, raw server responses, serving-host provenance and client batching metadata. Neither path measures TTFT; offline batch times and HTTP request latency have different meanings. The first warmup is excluded from measurement.

```bash
python -m pytest -q
```

Source code is independent of the upstream package. Vendored code remains under its original MIT license and the published artifact data under CC BY 4.0; the upstream license and dataset attribution are retained in `vendor/driftbench-ae`.
