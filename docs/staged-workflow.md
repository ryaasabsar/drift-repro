# Four stages and two transfers

Run inference on each accelerator, safety labeling on a fixed A100 environment,
then code and other scoring on RTX 3060. Choose comparisons after gathering the
scored runs. Inference is never repeated during transfer or evaluation.

Commands are run from the repository root. All hosts use the same checkout and
common `.venv-client`; serving interpreters are selected independently by the
suite. `scripts/results.sh` uses the client for inference, transfer and catalog,
and `.venv` for evaluation/comparison. `DRIFTBENCH_PYTHON` can explicitly override
this, but inference checks the common client pins. Avoid leaving an unrelated
serving interpreter in that variable.

## Inference host

```bash
bash scripts/bootstrap.sh --client-only
bash scripts/run_mi210.sh --run-id mi210-vllm-r01 --framework vllm
bash scripts/results.sh stage status results/runs/mi210-vllm-r01
bash scripts/results.sh handoff results/runs/mi210-vllm-r01 --to safety
```

Use `run_a100.sh` or `run_blackhole_p150b.sh` on those machines. P150b also needs
`--allow-experimental`. Omitting `--framework` runs every available framework;
filtering vLLM retains all three models and no pending SGLang settings.
`--limit 2` selects two inputs per workload for a small check. Full runs omit it.
Each filter/limit/seed/client-code change requires a new run ID. Resume unchanged
inference on its original host with the same command plus `--resume`.

## A100 safety host

Copy the one ZIP printed by `handoff` using SFTP, SCP, a shared folder or your
provider's file-transfer UI. No SSH access from this agent is required.

```bash
bash scripts/results.sh receive /path/to/INFERENCE.zip --output results/runs/mi210-for-safety
bash scripts/results.sh stage safety results/runs/mi210-for-safety
bash scripts/results.sh handoff results/runs/mi210-for-safety --to evaluate
```

For an A100 inference run, skip the first transfer and label its existing folder.
Safety uses the same pinned Llama-Guard-3-8B, BF16 CUDA, seed and deterministic
policy for every run. The inference server must have stopped to release memory.
The stage requires approved model access. Changing judge software/hardware/policy
must not mix labels within a run; comparisons check judge provenance.

## RTX code and final scoring host

Copy the newly generated safety ZIP to RTX:

```bash
bash scripts/results.sh receive /path/to/SAFETY.zip --output results/runs/mi210-for-eval
bash scripts/results.sh stage evaluate results/runs/mi210-for-eval
```

`evaluate` requires completed safety labels, executes generated HumanEval code
with Bubblewrap isolation on CPU, and scores math/long context using saved outputs.
It reuses valid code results after interruption. No large inference model or safety
judge is loaded here. The individual `stage code` and `stage final` commands remain
available. Chat has no single-run accuracy: its CPU embedding comparison happens
when two runs are compared with `--semantic`.

## Comparison host (RTX)

```bash
bash scripts/results.sh catalog results/runs
bash scripts/results.sh compare-many \
  --baseline results/runs/a100-for-eval/settings/a100_qwen25_7b_vllm \
  --candidates results/runs/mi210-for-eval/settings/mi210_qwen25_7b_vllm \
               results/runs/p150b-for-eval/settings/blackhole_p150b_qwen25_7b_vllm \
  --semantic --allow-confounded --output results/comparisons/qwen25
```

Use the actual paths printed by `catalog`. Any setting can be the baseline;
candidates can also be entire suite folders, in which case all their models and
frameworks are included. Source prompt sets must align. Multiple model/software/
hardware changes are explicit confounds; `--allow-confounded` acknowledges them
but does not relax source or evaluator consistency. Reports include per-pair
viewers plus aggregate CSV/JSON. Incompatible pairs are listed as errors and give
a nonzero exit status, while valid pairs remain saved.

## Portable folder and integrity

```text
results/runs/RUN/
  suite.json                       # Contains only settings selected for this run
  stages.json
  settings/SETTING/
    manifest.json                  # Immutable inference identity, client and server provenance
    config.json
    server.json
    startup-diagnostics.json       # Local diagnostic, not transferred
    code.jsonl                     # Generated code responses, not evaluation results
    math.jsonl
    safety.jsonl
    chat.jsonl
    long_context.jsonl
    safety-labels.jsonl             # Added on A100
    code-results.jsonl              # Added on RTX
    evaluation.json                # Final scores
    summary.csv
    tables/rows.csv
    datasets/                      # Bundled benchmark files, added during transfer
```

`handoff` embeds the checked `.tar.gz` and `.sha256` inside a single ZIP. `receive`
verifies the archive hash, file inventory, per-file hashes and inference/evaluation
fingerprints before publishing the imported directory. It never overwrites an
existing run or follows absolute paths saved on another machine. Only allowlisted
results and benchmark snapshots are transferred: no credentials, weights, caches,
environments or raw server logs. Keep diagnostics on the source host if debugging.
Checksums detect corruption; they are not sender authentication.

`receive` without `--output` chooses `results/runs/<ZIP-name>`. Each handoff has a
unique name and carries the latest labels forward. Older two-file `pack RUN
--output FILE.tar.gz` and `unpack FILE.tar.gz --output NEW_DIR` commands remain
supported unchanged. Imported runs can be evaluated and compared, but inference
resume is restricted to their original host/folder.

Old inference results remain readable. Their missing client-package provenance
is reported explicitly in comparisons. Older partially completed, unfiltered
suites must be completed or transferred as individual complete setting folders;
new filtered runs have no unselected pending entries.
