# Follow an experiment

Normal commands now report timestamped stages to **stderr**. The suite relays
its inference and evaluation subprocess events, so the terminal shows the
active setting and workload without needing to open each subprocess log.

```bash
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/my-settings --limit 2
```

The output identifies the setting's position in the selected list, model and
framework, then follows startup, prompt preparation, warmup, inference, server
shutdown, evaluation, table export and comparison. For example:

```text
[12:04:00Z] INFO    | [rtx3060_qwen35_08b_vllm] | suite | Setting started | position=1/4 | model=Qwen/Qwen3.5-0.8B | framework=vllm
[12:04:15Z] INFO    | [rtx3060_qwen35_08b_vllm] | startup | Waiting for server health check — still working | elapsed 15s | ...
[12:05:30Z] INFO    | [rtx3060_qwen35_08b_vllm] | inference/math | Responses saved | 4/10 (40.0%) | elapsed 18s | ... | workload_saved=2/2
```

Times use UTC (`Z`). Inference percentages count **saved responses across the
selected workloads**, including responses reused on resume. `workload_saved`
is the active workload's count. `batch` is the original one-based batch index;
resuming preserves batch membership. Batch duration covers generation; stage
elapsed time also includes saving and other work. Safety progress includes
reused judgments. Chat responses remain pairwise only until comparison.

Long operations emit a heartbeat every 15 seconds by default. During a long
batch, it shows the active workload, batch size and input length; single-request
batches also show the prompt ID. Safety evaluation identifies the prompt being
judged and the selected device. A heartbeat indicates that the operation is
still waiting; it does not assert that the model has generated more tokens.
Counts increase after responses or judgments are saved. Embedding counts
update after each side finishes encoding.

## Inspect from another terminal

```bash
source scripts/env.sh
python -m driftbench_runner status results/my-settings
python -m driftbench_runner status results/my-settings --watch
python -m driftbench_runner status results/my-settings --json
```

`status` accepts a suite directory or an individual setting directory. It shows
saved counts, evaluation counts, pending settings, recorded errors and the most
recent runner event with its age. It works before an inference manifest exists
and after copying a suite to another host. A recorded `running` status is not
proof that its process is still alive. `--watch` prints changes at the progress
interval; `--json --watch` emits one JSON object per changed snapshot. Ctrl-C
stops watching without stopping the experiment.

The existing command `python scripts/progress.py RUN_DIR` still produces JSON.
Add `--human` for readable output; it now supports suites and `--watch` too.

## Logs and verbosity

No extra dependency is required. Logs append on resume, and old child events
are not replayed to the console. Experimental manifests, response files and
scoring formats remain separate from logging.

```text
results/my-settings/
  logs/suite.log                    # readable events, including relayed child events
  logs/suite.events.jsonl           # the same events as structured records
  settings/SETTING/
    server.log                     # full serving-framework diagnostics
    launcher.log                   # launcher output and failures
    inference.log                  # full inference subprocess output
    evaluation.log                 # full evaluation subprocess output
    logs/launcher.events.jsonl
    logs/inference.events.jsonl
    logs/evaluation.events.jsonl
  comparisons/PAIR/
    comparison.log                 # full comparison subprocess output
    logs/comparison.events.jsonl
```

Standalone `run`, `score`, `evaluate`, `judge-safety`, `compare`, `collect` and
`export` commands append `logs/COMMAND.log` and `logs/COMMAND.events.jsonl` in
their result directory (or source run for evaluation/export). Standalone `serve`
uses the server metadata directory. The single-setting `run_served.sh` wrapper
also relays launcher events while waiting for readiness. Third-party output
from standalone engines or evaluator libraries may still appear on their
console; the managed suite keeps it in the subprocess logs, including comparisons.

These flags work before or after the subcommand:

```bash
# Quicker updates during an installation check:
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/quick-check --limit 2 --progress-interval 5

# Show only warnings/errors in the terminal; full runner events still go to files:
bash scripts/run_settings.sh --config suites/rtx3060.json \
  --output results/my-settings --resume --log-level warning
```

`--log-level debug` includes exception details in the terminal. With normal
verbosity, failures give a concise cause and the relevant log path; detailed
tracebacks remain in the saved runner logs. `--log-file PATH` and
`--events-file PATH` override the two runner log destinations. The suite passes
verbosity and interval to its children. Shell wrappers also accept
`DRIFTBENCH_LOG_LEVEL` and `DRIFTBENCH_PROGRESS_INTERVAL` environment variables.

Structured events contain `timestamp`, `level`, `pid`, `setting`, `stage` and
`message`, with relevant counts, workload, elapsed time and artifact paths.
Normal progress events do not include prompts, model responses or credentials;
raw framework/error logs may include library diagnostics. JSON output from
`preflight`, `doctor`, `suite --dry-run` and `status --json` stays on stdout, so
redirecting it to a file remains valid. Dry runs create no default log files.

## Validation

All [69 tests passed](../results/logging-tests.log), including subprocess event
forwarding, readable failures, partial-line handling, resume counts, JSON-only
stdout, comparison subprocesses and isolated code evaluation.

A fresh one-prompt Qwen3.5/vLLM run on the RTX 3060 exercised startup, inference,
shutdown, scoring and collection. The [suite log](../results/logging-validation-nvidia/logs/suite.log)
also records a completed resume that preserved the hashes and timestamps of all
five inference, evaluation and server evidence files. Its suite status is
`partial` because the other three settings were deliberately unselected.

Copies of ten existing responses covered all five evaluation workloads and a
managed semantic comparison. Both embedding sides reported 2/2 at completion;
source responses were unchanged. See the
[comparison log](../results/logging-validation-comparison-managed/logs/parent.log)
and [validation report](../results/logging-validation-report.json). No GPU compute
process or validation API endpoint remained running afterward.
