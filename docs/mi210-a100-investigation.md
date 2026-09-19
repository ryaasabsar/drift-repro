**Running the A100–MI210 investigation**

Entry point: `bash scripts/investigate.sh`. This implements the [investigation plan](mi210-a100-investigation-plan.md) using saved input token IDs, portable case bundles, fresh serving processes, normal DriftBench evaluation, and optional numerical probes. It does not modify existing benchmark runs or serving environments.

**Scope: math (GSM8K) and code (HumanEval) only.** Case selection, repeated inference, evaluation, and agreement reports cover these two workloads. This investigation requires no safety labeling or LlamaGuard. Accuracy and flip rates are reported separately for math and code.

**1. Prepare once on the machine containing the evaluated benchmark runs**

```bash
bash scripts/investigate.sh prepare \
  --output results/investigations/mi210-a100/cases-v1
```

Defaults: the existing A100-baseline inventory, Llama 3.2 1B and Qwen3.5 0.8B, eight regressions + eight improvements + eight unchanged controls per model. `triage.csv` covers code/math records for all five A100–MI210 model pairs. Review flags are hints, not automatic causal diagnoses. Copy this CSV elsewhere before adding manual classifications: bundle contents are hashed and immutable.

The prepared `cases-v1` folder is already available in this workspace. Do not run prepare again at that same path. To select a new bundle, choose another output name; `--models` accepts inventory model keys and `--per-group` changes the category size. The selection must contain both code and math and enough examples in every category.

Copy the **entire `cases-v1` directory** to both GPU machines, retaining the same relative path for these examples. It contains requests, references, original selected responses, both hardware configurations, and full dataset files reordered so that the selected records come first. Prompt content and source-record hashes are unchanged. The bundle can be moved; it no longer needs the original run directories.

**2. Preview and run on each allocated GPU host**

Run inside the allocated GPU job. Keep the scheduler's visible-device environment. Install/use the existing NVIDIA or ROCm serving environment; the common client remains separate. Commands are run from the repository root.

A100:

```bash
bash scripts/investigate.sh run \
  --bundle results/investigations/mi210-a100/cases-v1 \
  --device a100 --server-python .venv/bin/python \
  --conditions original serial --repeats 3 \
  --output results/investigations/mi210-a100/trials \
  --dry-run
```

Remove `--dry-run` after inspecting the generated trial configurations. MI210:

```bash
bash scripts/investigate.sh run \
  --bundle results/investigations/mi210-a100/cases-v1 \
  --device mi210 --server-python .venv-rocm-vllm/bin/python \
  --conditions original serial --repeats 3 \
  --output results/investigations/mi210-a100/trials
```

Each host runs 12 trials, generating 288 responses; both hosts together generate 576. To begin with only the repeated baseline, pass `--conditions original` (144 responses per host). `--models llama32_1b` restricts the model; `--port` changes the default local port 8001. Servers run sequentially; every trial starts and stops its own fresh server. Each trial performs the existing endpoint validation and short warmup, outside its scored responses.

| Condition | Change |
|---|---|
| `original` | Saved device-specific configuration, fixed selected-case order |
| `serial` | Original plus client batch size 1 and maximum running sequences 1 |
| `serial-eager` | Serial plus eager execution |
| `serial-eager-fp32` | Serial-eager plus FP32 dtype; optional and only where the model/backend supports it |

All conditions preserve the pinned checkpoint, tokenized input, seed, generation budget, and sampling parameters. These are diagnostic interventions, not claims of identical kernel implementations. FP32 can be slower or unsupported. Original subset trials preserve configuration but do **not** recreate the full benchmark's original batch membership or timing; record that limitation when comparing with old results.

Add `--resume` to skip completed, verified identical trials. It does not stitch partial generations from different server launches into one repeat. For a failed trial, retain its logs and use a new output root to retry, selecting the desired models/conditions. Do not combine duplicate `(device, model, condition, repeat)` identities in one report; choose one campaign deliberately.

**3. Move results and evaluate on one common host**

Trial layout:

```text
trials/
  a100/llama32_1b/original/repeat-001/
    experiment.json
    config.json, manifest.json, server.json
    code.jsonl, math.jsonl, datasets/
    launcher.log, server.log
  mi210/llama32_1b/original/repeat-001/
  ...
```

Copy the complete `trials/a100` and `trials/mi210` trees to your evaluation machine (for example the RTX3060 system with working bubblewrap). No weights, virtualenvs, cache directories, or credentials are needed for transfer. Plain directory/archive transfer is sufficient; internal absolute source paths are not followed when scoring the copied trials.

```bash
bash scripts/investigate.sh evaluate \
  results/investigations/mi210-a100/trials

bash scripts/investigate.sh report \
  --bundle results/investigations/mi210-a100/cases-v1 \
  --runs results/investigations/mi210-a100/trials \
  --output results/investigations/mi210-a100/report-v1
```

The evaluation uses existing sandboxed HumanEval and numeric math scoring. It never runs generated code directly on the GPU host. Missing bubblewrap leaves code evaluation incomplete; there is no unsandboxed fallback. `report` may run before evaluation to inspect token divergence, but missing/incomplete accuracy and flip rates stay null. Mixed evaluator protocols/environments are rejected. Use a new report directory when refreshing results.

If a campaign includes failed or unfinished trials, pass the completed `repeat-001`-style directories explicitly to `evaluate` and `report --runs`. Both commands reject incomplete trials in a supplied tree; keep those failed trial directories and their logs for diagnosis.

Report files:

- `accuracy.csv`: per-model/device/condition/repeat code and math accuracy, with scored counts.
- `agreement.csv`: within-device repeat comparisons, A100–MI210 comparisons, and adjacent condition comparisons; token agreement and correctness flips, with directions.
- `first-divergences.csv`: first differing token or termination position for each paired response.
- `probe-cases.json`: exact shared-prefix inputs for cross-device differences in repeat 1, separated by model/condition.
- `report.json`: explicit trial coverage, evaluation completeness, environment-difference flags, and interpretation limits.

These are deliberately selected cases, not new full-benchmark accuracy estimates. Cross-device reports include all repeat combinations, which are dependent observations; do not use their total count as independent statistical sample size. The report exposes available coverage and does not assume missing trials succeeded.

**4. Optional same-prefix top-k probes**

Copy `report-v1/probe-cases.json` to both accelerator hosts along with the same bundle. Example on A100:

```bash
bash scripts/investigate.sh probe \
  --bundle results/investigations/mi210-a100/cases-v1 \
  --requests results/investigations/mi210-a100/report-v1/probe-cases.json \
  --device a100 --model llama32_1b --condition serial \
  --server-python .venv/bin/python --top-k 10 \
  --output results/investigations/mi210-a100/probe-a100-llama-serial
```

On MI210 change `--device` to `mi210`, the serving Python to `.venv-rocm-vllm/bin/python`, and the output to `probe-mi210-llama-serial`. Then copy both probe directories to one machine:

```bash
bash scripts/investigate.sh compare-probes \
  results/investigations/mi210-a100/probe-a100-llama-serial/probe.json \
  results/investigations/mi210-a100/probe-mi210-llama-serial/probe.json \
  --output results/investigations/mi210-a100/topk-llama-serial.csv
```

The probe asks for one next token with top-k **log-probabilities identified by token ID**, using vLLM's completion API. It validates identical prefixes and refuses unsupported/malformed responses rather than fabricating token IDs or probabilities. It reports candidate overlap, top-1 agreement, margins, and error on shared candidates. These are not raw logits or full-vocabulary error. A full prefix is prefilled on a new server, so the diagnostic does not recreate the original incremental decode/cache state.

API fields are documented in [vLLM 0.17.1's completion API](https://docs.vllm.ai/en/v0.17.1/serving/openai_compatible_server/). A capability failure is retained in `probe.json`; its partial output cannot be treated as a complete comparison.

**5. Optional raw-logit/hidden-state reference captures**

This runs a separate, pinned Transformers eager forward pass on shared prefixes. It helps investigate precision, but **does not capture the original serving vLLM kernels**. Use a torch/Transformers environment supporting the chosen checkpoint. Start with Llama1B; the pinned Transformers 4.57.6 client may not recognize newer model architectures such as Qwen3.5. Unsupported models raise an error; the command does not upgrade your environment.

```bash
DRIFTBENCH_PYTHON=.venv/bin/python bash scripts/investigate.sh capture-reference \
  --bundle results/investigations/mi210-a100/cases-v1 \
  --requests results/investigations/mi210-a100/report-v1/probe-cases.json \
  --model llama32_1b --condition serial --device cuda \
  --dtype bfloat16 --layers 0 -1 --limit 3 \
  --output results/investigations/mi210-a100/capture-a100
```

On MI210 use `DRIFTBENCH_PYTHON=.venv-rocm-vllm/bin/python` and output `capture-mi210`. PyTorch's `cuda` device name also applies to ROCm. Choose explicit hidden-state indices to narrow down a layer boundary; index 0 is the first returned hidden state, not automatically the first transformer-block output.

```bash
bash scripts/investigate.sh compare-captures \
  results/investigations/mi210-a100/capture-a100 \
  results/investigations/mi210-a100/capture-mi210 \
  --output results/investigations/mi210-a100/reference-tensor-differences.json
```

Captures preserve model/revision, input tokens, original tensor representations, raw storage bits, and float32-expanded values. Comparison validates provenance and reports bitwise equality and numerical errors. Raw-logit arrays end in `_logits`; `__bits` arrays are storage only and are compared bitwise. Direct vLLM layer/kernel instrumentation remains follow-up work; do not claim a serving-kernel cause from this separate reference model alone.

**6. Optional synthetic precision microbenchmarks**

Run under each hardware's existing torch environment:

```bash
DRIFTBENCH_PYTHON=.venv/bin/python bash scripts/investigate.sh microbench \
  --device cuda --output results/investigations/mi210-a100/micro-a100

# On MI210:
DRIFTBENCH_PYTHON=.venv-rocm-vllm/bin/python bash scripts/investigate.sh microbench \
  --device cuda --output results/investigations/mi210-a100/micro-mi210
```

After moving both outputs to one host:

```bash
bash scripts/investigate.sh compare-microbench \
  results/investigations/mi210-a100/micro-a100 \
  results/investigations/mi210-a100/micro-mi210 \
  --output results/investigations/mi210-a100/micro-differences.json
```

The 16 fixed-input operations cover BF16 conversion near midpoints, zero-addition controls, addition, multiplication, scaled low-precision versus FP32 accumulation, late casts, and small matrix products. `microbench.json` records environment, input hashes, errors against a float64 reference, and implementation hash. The comparison verifies matching implementations, inputs, representations, and saved array hashes. Matmul inputs are exactly representable; conversion deliberately starts from FP32 values near BF16 boundaries. There is no claim that a particular matrix instruction or tensor-core path was used. `--device cpu` is for harness validation, not evidence about either GPU. `compare-tensors` is also available for arbitrary matching numeric NPZ arrays, but only the higher-level capture/microbench comparisons check their associated metadata.

**First-week stopping point**

Freeze and review the cases, run original/serial repeats, score them on one host, and inspect stable first-token disagreements. Add eager/FP32 trials or numerical probes only where those observations motivate them. Keep the original runs and all failed trial logs; classify each finding as generation variability, repeatable setup difference, extraction/scoring effect, or unresolved. Hardware root-cause attribution is not automatic.
