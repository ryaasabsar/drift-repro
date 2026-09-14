# Move inference results between evaluation machines

Use one run folder per experiment. Generate all five workloads on the accelerator,
judge safety on one fixed A100 environment, and execute HumanEval on the RTX 3060
system's CPU with Bubblewrap. Code and safety can run in either order. Final scoring
reads their saved results and needs neither a GPU nor Bubblewrap.

Run commands below from the project directory. `scripts/results.sh` uses `.venv/bin/python`;
set `DRIFTBENCH_PYTHON` to an **absolute Python executable path** to use another prepared
environment. The serving Python paths still come from the selected suite. All machines
need the same runner checkout/version; framework environments are needed only for inference,
and Transformers/PyTorch plus the approved judge model are needed on the safety host.
The code/final/transfer stages use the existing runner and standard Python libraries.

## 1. Save inference under a unique run ID

On A100:

```bash
bash scripts/results.sh infer \
  --config suites/a100-qwen25-7b-llamaguard3.json \
  --run-id a100-qwen25-7b-vllm-r01
```

This is **inference only**: no safety model or Bubblewrap check runs. Add `--limit 2`
for a ten-response smoke run, and give it a different run ID. Use `--dry-run` to inspect
the plan without downloads or accelerator access. Resume interrupted inference with the
same command plus `--resume`, on the original host and at the original output path.

For MI210 select `suites/mi210-qwen25-7b-llamaguard3.json`. For Blackhole select
`suites/blackhole-p150b-qwen25-7b-llamaguard3.json` and add `--allow-experimental`:
that model/board combination is still unverified and requires compatible TT kernels.
Existing RTX suites work through the same command. Choose a new run ID for every repeat
or changed setup; defaults go under `results/runs/` (`--output-root` changes that parent).

```text
results/runs/a100-qwen25-7b-vllm-r01/
  suite.json
  stages.json
  settings/a100_qwen25_7b_vllm/
    manifest.json             # Original inference identity and accelerator provenance
    config.json
    server.json
    code.jsonl                # Generated responses, not executed code
    math.jsonl
    safety.jsonl
    chat.jsonl
    long_context.jsonl
    safety-labels.jsonl        # Added by the safety stage
    code-results.jsonl         # Added by the code stage, with evaluator provenance
    evaluation.json           # Added by final scoring
    summary.csv
    tables/
```

A suite directory or an individual setting directory can be passed to every command
below. Suite commands discover settings relative to the supplied folder; saved absolute
paths from the inference host are preserved as historical metadata and are not used to
find responses on the next machine. Imported runs are for evaluation, not inference resume.

## 2. Run safety on the fixed evaluation host

If inference ran on A100, no transfer is needed yet:

```bash
bash scripts/results.sh stage safety results/runs/a100-qwen25-7b-vllm-r01
```

This defaults to the locked `meta-llama/Llama-Guard-3-8B` revision and GPU evaluation.
Run it after inference has stopped and released the GPU. For another pinned judge use
`--judge-model MODEL --judge-revision COMMIT`; `--judge-device cpu` is available, but
CPU and GPU judging have different precision/environment metadata. Use the **same model,
revision, precision, and evaluation environment for all experiments you intend to compare**.
The stage does not execute generated code.

For MI210/Blackhole inference, pack and transfer the run to A100 first using step 3.
Then run this same command on the unpacked directory. Rerunning a safety stage resumes
saved labels when outputs and judge environment match; mismatches are rejected.

## 3. Pack, move, and unpack

On the sending machine, after the current stage finishes:

```bash
bash scripts/results.sh pack results/runs/a100-qwen25-7b-vllm-r01 \
  --output results/transfers/a100-qwen25-7b-vllm-r01-safety.tar.gz
```

Move these two files together, for example through the host's file download interface
or as GitHub Release assets:

```text
a100-qwen25-7b-vllm-r01-safety.tar.gz
a100-qwen25-7b-vllm-r01-safety.tar.gz.sha256
```

The bundle contains manifests, responses, stage artifacts, and the **exact benchmark
files needed to score those responses**. No separate dataset download is required after
import. Packaging uses an explicit file allowlist: credentials, caches, model weights,
environments, logs and unrelated files are excluded. Derived tables are regenerated.
Keep the sending copy until the imported copy has been verified. Archives and checksums
are snapshots; export a new filename after each stage instead of replacing an old archive.

On the receiving RTX 3060 system:

```bash
bash scripts/results.sh unpack \
  results/transfers/a100-qwen25-7b-vllm-r01-safety.tar.gz \
  --output results/runs/a100-qwen25-7b-vllm-r01

bash scripts/results.sh stage status results/runs/a100-qwen25-7b-vllm-r01
```

Unpacking verifies the outer checksum, every included file, inference identities, and
available evaluation artifacts. It refuses an existing destination and unsafe archive
paths. Use a **new destination** when receiving a later stage of the same run, e.g.
`results/runs/a100-qwen25-7b-vllm-r01-after-code`. Renaming the outer folder does not
change the experiment identity. Checksum verification detects transfer corruption;
it is not an authenticity signature.

## 4. Execute code, then finalize

On the RTX 3060 system with a working Bubblewrap sandbox:

```bash
bash scripts/results.sh stage code results/runs/a100-qwen25-7b-vllm-r01
bash scripts/results.sh stage final results/runs/a100-qwen25-7b-vllm-r01
```

`code` evaluates only HumanEval on CPU. It appends one result per response, resumes missing
responses, and preserves detailed Bubblewrap errors. Saved scores retain the code evaluator
version, execution environment, and source/output hashes. A completed code stage can be
reused without executing code again.

`final` requires all applicable code and safety results. It incorporates those results,
scores math and long context, writes `evaluation.json`, `summary.csv`, and per-response
tables, and updates stage/suite status. It never loads a judge or executes generated code.
Chat remains `pairwise_only` until two runs are compared; this is expected, not a missing
stage. Original inference responses, timings, model settings, and accelerator provenance
remain unchanged.

You may instead execute code first, pack that result, then unpack it on A100 for safety
and final scoring. Carry the **latest complete folder forward** through the stages. Do not
unpack two independently evaluated copies over each other; this workflow deliberately
avoids implicit merging or overwriting of results.

## 5. Compare evaluated settings

On your analysis machine, use the setting directories (not the suite root):

```bash
bash scripts/results.sh compare \
  results/runs/a100-qwen25-7b-vllm-r01/settings/a100_qwen25_7b_vllm \
  results/runs/mi210-qwen25-7b-vllm-r01/settings/mi210_qwen25_7b_vllm \
  --semantic --allow-confounded \
  --output results/comparisons/a100-vs-mi210-qwen25-7b
```

The example labels the CUDA/ROCm software-stack differences explicitly with
`--allow-confounded`; it is a comparison of complete setups, not proof of a hardware-only
effect. This flag does not waive evaluator compatibility checks. Existing RTX results
from Qwen3.5-0.8B are a different-model experiment; use matching inference models and
controls when you want to study hardware/framework drift. Semantic comparison uses the
locked embedding model on CPU and may need its first download.

## Status and reruns

`stage status FOLDER` derives completion from verified files instead of trusting an old
status flag. `stages.json` is refreshed by stage commands and import. Finalized suite
entries become `complete`, and their old inference plan remains unchanged. An interrupted
stage is rerun using the same command; already saved compatible results are reused.
Do not run two operations against the same folder at once: the commands acquire the
same run/suite locks used by inference.

Old result folders produced by `run_a100.sh --inference-only` can also be packed and
staged. Keep matching benchmark files in the source checkout for the initial pack;
thereafter they travel inside the bundle. Existing one-shot `score`/`evaluate` commands
remain available, but use `stage` consistently for this portable workflow.
