# Run settings, gather responses, compare rows

Each setting has one row per generated response, including its workload, stable
prompt ID, full response, result and score. Compare using `(workload, prompt_id)`;
spreadsheet row numbers are only a display order. All commands below run from
the project root after the environments in [the platform guide](platforms.md)
are installed.

The shell wrapper uses `.venv` for the benchmark client and evaluator. On a
different host, this can be a client-only environment; each setting names its
separate serving Python. To use another client environment directly, run its
Python with `-m driftbench_runner suite` and the same arguments.

## Run all selected settings on one GPU

```bash
# First inspect settings and input counts; this does not load a model.
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/my-3060-settings --dry-run

# Full experiment: 2,284 prompts per setting, all five workloads.
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/my-3060-settings

# After interruption, use the same selection and output directory.
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/my-3060-settings --resume
```

For a quick installation check, use a **different output directory** and add
`--limit 2` (10 prompts per setting). Keep that limit when resuming. A limited run
cannot be expanded into a full run in the same directory: batch membership and
the selected inputs are experimental controls.

The script starts one server, runs inference, shuts down that server, evaluates
on the client environment, exports tables, and moves to the next setting.
Its console follows each stage and displays workload progress. Use
`python -m driftbench_runner status results/my-3060-settings --watch` from
another terminal, or read `logs/suite.log` in the output directory. See the
[logging guide](logging.md) for verbosity, heartbeat intervals and JSON events.
The default safety judge is the pinned Qwen3Guard-Gen-0.6B on CPU. HumanEval uses
isolated execution. GPU experiments launched by the convenience scripts share a
workspace lock. Servers started manually are outside that lock.

The NVIDIA preset contains:

| Setting ID | Model | Framework |
|---|---|---|
| `rtx3060_qwen35_08b_vllm` | Qwen3.5-0.8B | vLLM |
| `rtx3060_qwen35_08b_sglang` | Qwen3.5-0.8B | SGLang |
| `rtx3060_qwen25_05b_vllm` | Qwen2.5-0.5B-Instruct | vLLM |
| `rtx3060_qwen25_05b_tensorrt` | Qwen2.5-0.5B-Instruct | TensorRT-LLM |

The full preset therefore runs 9,136 inference requests. CPU safety evaluation
can take substantial time. On one common NVIDIA evaluation host, you can set
`evaluation.judge_device` to `cuda` for the smaller judge after each server has
stopped. Use the same judge device for both sides of a comparison and a new
output directory when changing this control.

TensorRT-LLM 1.2.1 in this workspace uses Qwen2.5-0.5B-Instruct because its installed model
registry lacks Qwen3.5. The two configured comparisons pair the **same model**
across frameworks. Engine and package differences are declared in the reports.

Qwen2.5's smaller KV cache also leaves room for the long-context workload on
this 6 GB GPU. The TensorRT preset caps its KV pool at 65,536 tokens, uses BF16,
and requests a 32,768-token context. The client checks the observed server limit
before inference. Its paired vLLM preset uses the same
checkpoint and prompts. Qwen3-0.6B profiles and earlier results remain available,
but full-length TensorRT runs with that model proved sensitive to available
VRAM. The Qwen2.5 architecture is recorded in its
[official configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json).

To run just some settings, supply their IDs:

```bash
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/qwen35-settings \
  --settings rtx3060_qwen35_08b_vllm rtx3060_qwen35_08b_sglang
```

The suite status is `partial` when other settings in the preset are unselected.
Both selected settings still have their own completion status and comparison.
Later, `--resume` without `--settings` runs the remaining settings and reuses
completed inference. `--workloads code math` selects benchmarks; `--device 0`
sets the visible NVIDIA/AMD accelerator. Keep these values unchanged on resume.
`--continue-on-error` attempts other settings after a failure and still returns
a failing exit status. Failed or pending evaluations remain visible.

## Change the list of settings

Copy an existing suite JSON and edit its `settings` list. Each item contains:

```json
{
  "id": "gpu_model_framework",
  "config": "../configs/your-profile.json",
  "server_python": "../.venv-framework/bin/python"
}
```

Paths in this list are relative to the **suite JSON's directory**. Use unique
setting IDs. The script copies each profile into that setting's output folder,
sets its `setup_id`, and gives HTTP servers separate logs and metadata files.
Models and judges must use pinned Hub revisions. A suite runs on one host/vendor;
run another suite on another machine, then bring the saved folders together.

Prepared hardware templates are `suites/rx7900xt.json`, `suites/mi210.json`,
`suites/wormhole-n150.json`, `suites/wormhole-n300.json`, and
`suites/blackhole-p300.json`. Their ROCm/TT Python paths are placeholders for
environments installed on those hosts. They have **not** been tested on physical
AMD or TT devices. The TT templates specify concrete boards and supported-model
examples; adjust them to the board and vendor environment you actually have.

For later evaluation on one common machine, run a suite with `--inference-only`,
copy its complete setting directories to that machine, then use `score` below.
The reserved LlamaGuard slot is `evaluation.judge_model` with
`meta-llama/Llama-Guard-3-8B` and an approved pinned `judge_revision`. It requires
model access and a suitable evaluation host.

## Where the rows are saved

```text
results/my-3060-settings/
  suite.json                        # plan, progress and failures
  settings/gpu_model_framework/
    config.json, manifest.json      # exact settings and observed environment
    code.jsonl, math.jsonl, ...     # original responses and token traces
    safety-labels.jsonl             # judge output and provenance
    evaluation.json, summary.csv
    tables/
      rows.csv, rows.jsonl          # all this setting's responses
      code.csv, math.csv, safety.csv, chat.csv, long_context.csv
      setting.json
  collection/
    rows.csv, rows.jsonl            # all settings, one response per row
    matrix.csv                     # one prompt per row, columns for every setting
    workloads/code.csv, math.csv, safety.csv, chat.csv, long_context.csv
    settings.csv, settings.json    # counts, paths, configurations and provenance
  comparisons/baseline__vs__candidate/
    rows.html                      # searchable, standalone side-by-side viewer
    rows.csv, rows.jsonl            # paired responses, scores and changes
    workloads/code.csv, math.csv, safety.csv, chat.csv, long_context.csv
    comparison.csv, comparison.json
```

Open `collection/workloads/math.csv`, for example, to see columns like:

| row_id | setting_A.result | setting_B.result | setting_A.output_text | setting_B.output_text |
|---|---|---|---|---|
| math:example_1 | correct | incorrect | full response A | full response B |

This row is illustrative. Actual files retain the authors' prompt IDs and full
multiline responses. CSV and JSONL contain the same response text; use JSONL for
typed booleans, nulls and structured judge metadata.

Results have workload-specific meanings:

| Workload / benchmark | `result` | Evaluation |
|---|---|---|
| code / HumanEval | correct or incorrect | Executed tests, pass@1 |
| math / GSM8K | correct or incorrect | Final numeric exact match |
| safety / AdvBench | safe or unsafe | Small safety judge; severity retained |
| chat / LMSYS-Chat-1M | pairwise_only | Compare embeddings against another setting |
| long_context / LongBench Qasper | correct or incorrect | Token F1; threshold 0.5 |

`pending`, `error`, and `missing` are separate states. They never count as
incorrect answers. Safety's boolean `correct` means “classified safe,” not task
accuracy. Chat has no fabricated correct/incorrect label. Its comparison rows
contain cosine shift and the substantial-drift flag when `--semantic` is used.

## Gather existing or future runs

Existing saved runs work without repeating inference:

```bash
source scripts/env.sh
python -m driftbench_runner export results/rtx3060-qwen35-vllm

python -m driftbench_runner collect \
  results/rtx3060-qwen35-vllm results/a100-qwen35-vllm \
  --output results/gathered-settings
```

`collect` accepts different subsets and puts `missing` in matrix cells without a
response. It rejects conflicting source hashes. Repeated setting IDs require
unique aliases via `--aliases aliases.json`, a JSON object mapping the supplied
run paths to your chosen names. Collection gathers observations; it does not
assert that the settings or their evaluators are comparable.

Copy **complete setting directories**, including original JSONL, manifest,
evaluation and judge labels, to preserve the evidence for future comparisons.
An exported CSV is convenient for analysis; the original directory is needed
for verified drift reports. Gathering and comparison need no running server.

To evaluate an inference-only setting on a common host:

```bash
python -m driftbench_runner score results/a100-qwen35-vllm
```

If old safety labels used another evaluation device or judge, choose a new
`--labels-output results/a100-qwen35-vllm/safety-common-host.jsonl`. Evaluate
both runs using the same model, revision and evaluation environment. `score`
updates evaluation files and tables while preserving inference responses.

## Compare two settings row by row

```bash
python -m driftbench_runner compare \
  results/my-3060-settings/settings/rtx3060_qwen35_08b_vllm \
  results/my-3060-settings/settings/rtx3060_qwen35_08b_sglang \
  --semantic --allow-confounded --output results/vllm-vs-sglang
```

Open `results/vllm-vs-sglang/rows.html`. Filter by workload, label flips, changed
responses or prompt ID; expand a row to inspect both complete answers. The HTML
is standalone and also works after copying it to another machine.

Comparison requires complete runs with **identical prompt-ID sets**, matching
sources and compatible evaluators. It refuses stale scores. `--allow-confounded`
acknowledges recorded changes to experimental controls; it does not bypass
source or evaluator checks. For a controlled RTX 3060 versus A100 comparison,
keep model, framework version, precision, prompting, decoding, batches and
transport fixed, and omit this flag unless you intend to report extra changes.

Text changes, correctness/safety flips and chat semantic changes are separate
columns. Missing generated token IDs produce an unavailable token-drift value,
not a zero. See [methodology](methodology.md) for departures from the paper.
