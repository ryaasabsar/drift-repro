# DriftBench across accelerators

Run the five published DriftBench workloads, save every response, and compare
outputs and evaluated drift across hardware and serving frameworks.

| Inference host | Frameworks | Suite | Settings |
|---|---|---|---:|
| NVIDIA A100 40 GB | vLLM, SGLang | `suites/a100.json` | 6 |
| AMD Instinct MI210 | vLLM, SGLang | `suites/mi210.json` | 6 |
| Tenstorrent Blackhole P150b | vLLM with TT plugin | `suites/blackhole-p150b.json` | 3 |

Every suite uses these exact models:

| Model | Hugging Face checkpoint | Prompt format |
|---|---|---|
| Qwen3.5 9B Base | `Qwen/Qwen3.5-9B-Base` | Raw published prompt |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | Official chat template |
| Llama3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Official chat template |

Qwen3.5 **9B Base** is the confirmed model; there is no substitution with Qwen3-8B.
Model revisions, prompt formats, seed 42, greedy decoding, 512 output tokens,
32K requested context and one request at a time are explicit in `configs/`.
Sampling penalties are neutral; SGLang requests its deterministic inference mode.
This favors repeatability over throughput. Start new run IDs for these settings;
previous batches of 16 are a different experimental configuration.
The same model uses the same prompts and sampling controls across settings.
Internal kernels, precision and serving versions can differ and are recorded.

A100 vLLM/SGLang operation was reported working by the user. MI210 and P150b
configs require compatible vendor runtimes; this refactor does not claim new
hardware validation. In particular, P150b model kernels and context capacity
still need validation on the actual machine. See [setup](docs/setup.md).

## Run inference

Keep the working A100 environments. The default server Python paths are:

| Host | vLLM | SGLang |
|---|---|---|
| A100 | `.venv/bin/python` | `.venv-sglang/bin/python` |
| MI210 | `.venv-rocm-vllm/bin/python` | `.venv-rocm-sglang/bin/python` |
| P150b | `.venv-tt-vllm/bin/python` | Not used |

From the project root on each inference machine:

```bash
# Run the command for that machine; use a new ID for each experiment.
bash scripts/run_a100.sh --run-id a100-r01
bash scripts/run_mi210.sh --run-id mi210-r01
bash scripts/run_blackhole_p150b.sh --run-id p150b-r01 --allow-experimental
```

These scripts run **inference only**. They start one server at a time, save all
responses, stop the server, and advance to the next setting. They do not load
LlamaGuard or execute generated code. All five workloads are included; no limit
means all 2,284 published inputs per setting. `--dry-run` inspects the plan;
`--resume` resumes the same unchanged run on its original host. Changed configs
require a new run ID. Small checks and historical RTX experiments live in `.archive`.

The common entry point is equivalent:

```bash
bash scripts/results.sh infer --config suites/a100.json --run-id a100-r01
```

## Score on the designated machines

The workflow is **inference host → A100 safety judge → RTX evaluation**.
Carry the latest run folder forward; no inference is repeated during scoring.

1. On MI210 or P150b, pack the inference run and move the archive **and its
   `.sha256` file** to A100. For A100 inference, use its existing run folder.
2. On A100, unpack into a new directory and run `stage safety`. It defaults to
   the pinned `meta-llama/Llama-Guard-3-8B`, BF16 on CUDA. Use the same A100 judge
   environment for every run being compared.
3. Pack the safety-scored run, move both transfer files to the RTX machine,
   unpack, then run `stage code` and `stage final` there.

```bash
# Sending machine (example: MI210)
bash scripts/results.sh pack results/runs/mi210-r01 \
  --output results/transfers/mi210-r01-inference.tar.gz

# A100, after copying the two transfer files here
bash scripts/results.sh unpack results/transfers/mi210-r01-inference.tar.gz \
  --output results/runs/mi210-r01-for-safety
bash scripts/results.sh stage safety results/runs/mi210-r01-for-safety
bash scripts/results.sh pack results/runs/mi210-r01-for-safety \
  --output results/transfers/mi210-r01-safety.tar.gz

# RTX machine, after copying the new archive and checksum here
bash scripts/results.sh unpack results/transfers/mi210-r01-safety.tar.gz \
  --output results/runs/mi210-r01-for-eval
bash scripts/results.sh stage code results/runs/mi210-r01-for-eval
bash scripts/results.sh stage final results/runs/mi210-r01-for-eval
```

Safety requires the prepared A100 Python environment. HumanEval uses the RTX
machine's CPU with Bubblewrap isolation; the 7–9B inference models and 8B judge
are never loaded on its 6 GB GPU. Final scoring reads the saved evaluations and
scores math/long context. [Detailed stages and transfers](docs/staged-workflow.md).

## Results and comparisons

A run contains `settings/<setting-id>/`, with the immutable `manifest.json`, one
response JSONL per workload, safety labels, code results, final `evaluation.json`,
`summary.csv`, and `tables/rows.csv`. The suite's `collection/` joins settings by
workload and prompt ID. Transfer bundles include exact benchmark files and
provenance, exclude credentials/model caches, and verify file checksums on import.

| Workload | Inputs | Evaluation |
|---|---:|---|
| HumanEval code | 164 | Isolated execution, pass@1 |
| GSM8K math | 500 | Final-number exact match |
| AdvBench safety | 520 | Llama-Guard-3-8B response classification on A100 |
| LMSYS chat | 1,000 | Paired embedding cosine shift |
| LongBench Qasper | 100 | Token F1; correctness threshold 0.5 |

```bash
bash scripts/results.sh compare \
  results/runs/a100-r01-for-eval/settings/a100_qwen25_7b_vllm \
  results/runs/mi210-r01-for-eval/settings/mi210_qwen25_7b_vllm \
  --semantic --allow-confounded --output results/comparisons/a100-mi210-qwen25
```

Compare matching models and input sets after common evaluation. The report
separates text changes from correctness/safety label flips and records changes
in hardware, software and engine settings. `--allow-confounded` labels comparisons
with multiple changed controls; it does not bypass evaluator compatibility.
Chat is `pairwise_only` until compared. Semantic comparison uses the pinned
embedding model on CPU.

This adapts [DriftBench](https://openreview.net/forum?id=Xfzzp6grRP) using its
[published artifact](https://github.com/GianluigiVitale/driftbench-ae). The artifact
publishes 1,000 chat inputs whereas the paper reports 973; all published inputs
are retained here. [Methodology](docs/methodology.md) explains this deviation.

`docs/` contains current setup, stages, methodology and logging guidance.
`.archive/legacy/` preserves historical profiles, TensorRT/other boards, reports
and tests; it is not the production entry point. `.archive/checks/` contains only
checks for this refactor, executed locally on the RTX workspace. Existing results,
model caches and installed environments are retained.
