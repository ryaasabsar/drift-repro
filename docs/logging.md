# Follow an experiment

Normal commands now report timestamped stages to **stderr**. The suite relays
its serving and inference subprocess events, so the terminal shows the
active setting and workload without needing to open each subprocess log.

```bash
bash scripts/results.sh infer --config suites/a100.json \
  --run-id a100-r01
```

The output identifies the setting's position in the selected list, model and
framework, then follows startup, prompt preparation, warmup, inference, server
shutdown and table export. The separate stage and compare commands report their own progress. For example:

```text
[12:04:00Z] INFO    | [a100_qwen35_9b_base_vllm] | suite | Setting started | position=1/6 | model=Qwen/Qwen3.5-9B-Base | framework=vllm
[12:04:15Z] INFO    | [a100_qwen35_9b_base_vllm] | startup | Waiting for server health check — still working | elapsed 15s | ...
[12:05:30Z] INFO    | [a100_qwen35_9b_base_vllm] | inference/math | Responses saved | 4/10 (40.0%) | elapsed 18s | ... | workload_saved=2/2
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
python -m driftbench_runner status results/runs/a100-r01
python -m driftbench_runner status results/runs/a100-r01 --watch
python -m driftbench_runner status results/runs/a100-r01 --json
```

`status` accepts a suite directory or an individual setting directory. It shows
saved counts, evaluation counts, pending settings, recorded errors and the most
recent runner event with its age. It works before an inference manifest exists
and after copying a suite to another host. A recorded `running` status is not
proof that its process is still alive. `--watch` prints changes at the progress
interval; `--json --watch` emits one JSON object per changed snapshot. Ctrl-C
stops watching without stopping the experiment.

Use `bash scripts/results.sh status RUN_DIR --json` for JSON output.
Omit `--json` for readable output; `--watch` follows changes.

## Logs and verbosity

No extra dependency is required. Logs append on resume, and old child events
are not replayed to the console. Experimental manifests, response files and
scoring formats remain separate from logging.

```text
results/runs/a100-r01/
  logs/suite.log                    # readable events, including relayed child events
  logs/suite.events.jsonl           # the same events as structured records
  settings/SETTING/
    server.log                     # full serving-framework diagnostics
    launcher.log                   # launcher output and failures
    inference.log                  # full inference subprocess output
    logs/launcher.events.jsonl
    logs/inference.events.jsonl
  comparisons/PAIR/
    comparison.log                 # full comparison subprocess output
    logs/comparison.events.jsonl
```

Standalone `run`, `stage`, `compare`, `collect` and `export` commands append logs in their output or run directory. `serve` uses the server metadata directory. Raw serving output is saved in each setting's `server.log`.

These flags work before or after the subcommand:

```bash
# Quicker updates during an installation check:
bash scripts/results.sh infer --config suites/a100.json \
  --run-id a100-r01 --progress-interval 5

# Show only warnings/errors in the terminal; full runner events still go to files:
bash scripts/results.sh infer --config suites/a100.json \
  --run-id a100-r01 --resume --log-level warning
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
