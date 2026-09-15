# DriftBench across accelerators

The workflow is **inference on each accelerator → safety labels on A100 → other
evaluation on RTX 3060 → paired comparisons**. Every response and its experimental
controls remain in a portable run folder.

| Inference host | Frameworks | Suite |
|---|---|---|
| NVIDIA A100 40 GB | vLLM, SGLang | `suites/a100.json` |
| AMD MI210 | vLLM, SGLang | `suites/mi210.json` |
| Tenstorrent Blackhole P150b | TT vLLM | `suites/blackhole-p150b.json` |

Each framework runs **Qwen3.5-9B-Base**, **Qwen2.5-7B-Instruct** and
**Llama-3.1-8B-Instruct**, with pinned revisions. Base uses raw published prompts;
Instruct models use their official chat templates. Seed 42, temperature 0,
neutral sampling penalties, 512 output tokens and one request at a time are
consistent across settings. Identical output across hardware is not guaranteed.

## 1. Prepare and run inference

Use the same repository commit on every host. Prepare the **same tokenizer client**
on every machine, including machines with existing serving environments:

```bash
bash scripts/bootstrap.sh --client-only
```

This installs Python 3.12.14 and `requirements.client.lock.txt` into `.venv-client`.
It does not install Torch, accelerator drivers or Bubblewrap. Serving processes
use the separate Python paths in each suite. See [host setup](docs/setup.md).

Choose the command for the inference machine:

```bash
bash scripts/run_a100.sh --run-id a100-vllm-r01 --framework vllm
bash scripts/run_mi210.sh --run-id mi210-vllm-r01 --framework vllm
bash scripts/run_blackhole_p150b.sh --run-id p150b-vllm-r01 --framework vllm --allow-experimental
```

`--framework vllm` runs vLLM on **all three models**. Use `--framework sglang` on
A100/MI210, or omit the filter to run every framework available in that suite.
`--settings ID...` optionally narrows the selection further. Filtered runs contain
only the selected settings, so they can be evaluated and transferred as complete runs.

Add `--limit 2` and a separate run ID for a small check; omit it for the complete
2,284 inputs per setting. `--dry-run` prints the selected plan. To resume, use
the same command, filters and run ID plus `--resume` on the original host.
Changed filters, client code or experimental controls require a new run ID.
Inference never loads the safety judge or executes generated code.

For separate small-model verification, use `Qwen/Qwen3.5-0.8B` (chat, thinking
disabled) and `meta-llama/Llama-3.2-1B-Instruct` with the existing host wrappers:

```bash
bash scripts/run_a100.sh --config suites/a100-small.json --framework vllm --limit 2 --run-id a100-small-smoke
bash scripts/run_mi210.sh --config suites/mi210-small.json --framework vllm --limit 2 --run-id mi210-small-smoke
bash scripts/run_blackhole_p150b.sh --config suites/blackhole-p150b-small.json --framework vllm --limit 2 --run-id p150b-small-smoke --allow-experimental
```

Each command selects two models and five workloads: 20 responses per host.
Use `--framework sglang` on NVIDIA/AMD once that serving environment is installed.
Omit `--limit` for all prompts. These suites retain the same generation controls
and Llama Guard 3-8B evaluation; the larger-model suites remain available.
Llama 3.2 requires its own approved model access. The P150b profiles are
experimental: architecture registration does not establish support for these
exact checkpoints or the configured context length.

**Comparability:** the tokenizer client, prompts and sampling controls match.
Serving dependencies cannot all be identical across vendors/frameworks:
NVIDIA/AMD target the same upstream releases, while the pinned TT plugin needs
vLLM 0.25.1 instead of 0.17.1. SGLang and vLLM also require different Torch and
Transformers versions. `runtime-contracts.json` declares these differences;
startup checks the core versions, and manifests record all installed package
versions, client versions, driver information and runner source identity.
Do not describe those cross-stack comparisons as hardware-only effects.

Before a large run, check the actual serving interpreters inside your allocated job:

```bash
bash scripts/results.sh doctor --suite suites/a100.json --framework vllm sglang
```

A100 with driver 570.195.03 uses the pinned CUDA 12.8 builds. The runner tests a
real CUDA calculation in a fresh process, retries a failed probe once, and saves
`startup-diagnostics.json`. It preserves scheduler device visibility and never
resets GPUs. [CUDA troubleshooting](docs/setup.md#cuda-detection-on-a100).

## 2. Label safety on A100

For inference already performed on A100, use its existing run directory. For
MI210/P150b, create a one-file transfer:

```bash
# Inference host: the command prints the generated ZIP path.
bash scripts/results.sh handoff results/runs/mi210-vllm-r01 --to safety
```

Copy that **single ZIP** to A100 using your available file-transfer method, then:

```bash
bash scripts/results.sh receive /path/to/TRANSFER.zip --output results/runs/mi210-for-safety
bash scripts/results.sh stage safety results/runs/mi210-for-safety
bash scripts/results.sh handoff results/runs/mi210-for-safety --to evaluate
```

Safety uses the pinned **Llama-Guard-3-8B**, BF16 on CUDA, in the same A100 `.venv`
for every inference host. Approved Hugging Face access is required. Run it after
inference releases the GPU. The transfer includes both the archive and checksum
inside the ZIP; credentials, model weights and environments are excluded.

## 3. Evaluate on RTX 3060

Copy the second ZIP to RTX, then:

```bash
bash scripts/results.sh receive /path/to/SAFETY_TRANSFER.zip --output results/runs/mi210-for-eval
bash scripts/results.sh stage evaluate results/runs/mi210-for-eval
```

`evaluate` checks safety labels, executes HumanEval using **CPU + Bubblewrap**,
then scores math and long context and exports tables. It does not load the 8B
safety judge. It resumes saved code results. `stage code` and `stage final` remain
available separately. Chat is evaluated pairwise during semantic comparisons.
Repeat these steps for each inference run. [Folder layout and transfers](docs/staged-workflow.md).

## 4. Choose any baseline and compare

List available setting paths on RTX:

```bash
bash scripts/results.sh catalog results/runs
```

Choose **one setting directory** as the baseline. Candidates can be individual
settings or whole imported suite folders, including other frameworks, models,
hardware or repeat runs:

```bash
bash scripts/results.sh compare-many \
  --baseline results/runs/a100-for-eval/settings/a100_qwen25_7b_vllm \
  --candidates results/runs/mi210-for-eval results/runs/p150b-for-eval \
  --semantic --allow-confounded \
  --output results/comparisons/qwen25-a100-vs-others
```

This compares the baseline with every candidate setting, including different
models. To compare only matching models, supply their individual setting paths.
`--dry-run` lists the pairs. Results include `comparisons.json`, an aggregate
`comparisons.csv`, and per-pair reports and `rows.html` viewers. Incompatible input
sets or evaluators are reported explicitly and produce a nonzero exit status;
they are never counted as zero drift.

`--allow-confounded` permits and labels changed model/software/engine controls;
it never bypasses source-prompt or evaluator checks. Omit it when testing a strict
comparison. Token-sequence comparisons require the same model/revision. The
original two-directory `compare BASELINE CANDIDATE --output ...` command remains.

| Workload | Published inputs | Evaluation |
|---|---:|---|
| HumanEval code | 164 | Isolated execution, pass@1 |
| GSM8K math | 500 | Final-number exact match |
| AdvBench safety | 520 | Llama-Guard-3-8B on A100 |
| LMSYS chat | 1,000 | Paired CPU embedding cosine shift |
| LongBench Qasper | 100 | Token F1, correctness threshold 0.5 |

This adapts [DriftBench](https://openreview.net/forum?id=Xfzzp6grRP) and its
[published artifact](https://github.com/GianluigiVitale/driftbench-ae). The artifact
has 1,000 chat inputs whereas the paper reports 973; all published inputs are
retained. See [methodology](docs/methodology.md) for differences and limitations.

The production 7–9B models still need validation on each target runtime, particularly
MI210 and P150b. Local checks use the RTX machine and a cached small model.
Focused regression checks live in `.archive/checks`; generated test artifacts and
historical experiments remain ignored under `.archive`.
