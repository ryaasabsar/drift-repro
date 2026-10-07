# Week 1 (reworked): does the answer change across devices, and is it repeatable?

One model, one fixed case set, batch size 1, three fresh servers per device.
The goal is a clean starting point for the top-down study: benchmark answers →
first divergent token → (Week 2) the layer and operator behind it.

## What changed from the original Week 1, and why

| Original | Rework | Reason |
|---|---|---|
| Qwen3.5-0.8B first | **Llama-3.2-1B-Instruct only** | Qwen3.5 was not repeatable on A100 even in `serial` (58% token identity, 36% flips between repeats on code), so its cross-device differences can't be separated from noise. Llama in serial mode was 100% repeatable on both GPUs and still differed across them. |
| 24 hand-picked cases (8 regressions + 8 improvements + 8 controls per model) | **All 164 HumanEval + the full GSM8K test split (1319)** = 1483 cases | 24 cases selected *because* they flipped can't estimate accuracy or drift rates, and they bias toward flips. The full sets give standard benchmark accuracy with no sampling choice to defend. The old DriftBench 500 GSM8K prompts are a subset of these 1319. |
| `original` + `serial` conditions | **`serial` only** (`--max-num-seqs 1`, one request at a time, prefix caching off) | Batching made `original` non-repeatable; serial is the controlled condition. |
| Separate top-k probe step | **Top-5 log-probs saved during generation** (GPUs) | The margin between the top two candidates at the first divergent token comes for free, because both devices share the exact prefix up to that point. |
| Flip counts only | **Four outcome counts + exact McNemar test + Wilson CIs** | Separates "answers change" from "one device is more accurate". |
| Bundle + repo runner on each host | **Frozen token IDs in `data/cases.jsonl`, stdlib-only client** | Every host sends byte-identical inputs and needs no tokenizer. The token IDs are those used in the earlier a100/mi210/p150b runs (verified identical on all three). |

The vLLM settings are copied from `configs/*-llama32-1b-instruct-vllm-http.json`
(same revision, dtype, seed, sampling, context length). The only changes are
`--max-num-seqs 1` and prefix caching off. Scoring reuses DriftBench's own
`extract_number` and sandboxed `execute_code`, so accuracy is comparable with
earlier runs.

## Files

| File | Purpose |
|---|---|
| `run.sh` | Per device: start a fresh vLLM server, generate all cases, stop it; repeat |
| `week1.py` | `generate` (called by `run.sh`), `score`, `report` |
| `data/cases.jsonl` | 1483 frozen cases (sha256 `c87bd3e7…76b8`) |
| `data/build_cases.py` | How `cases.jsonl` was made (see below); not needed to run the experiment |

## 1. Generate on each accelerator

From the repository root, inside the allocated job:

```bash
bash normbench-rework/run.sh a100    # NVIDIA, uses .venv
bash normbench-rework/run.sh mi210   # AMD, uses .venv-rocm-vllm
bash normbench-rework/run.sh p150b   # Tenstorrent, uses .venv-tt-vllm (+ activate-tt.sh)
```

Each writes `normbench-rework/results/<device>/r1..r3/` (outputs, server log,
exact server command, package versions). The defaults are 3 repeats and port 8001.
Options: `--repeats N`, `--out DIR`, `--port P`, and `--limit N` for a quick
smoke test (first N cases of each workload). `--percent P` runs a spread-out subset
(every round(100/P)-th case of each workload: 10 → 149 cases, 20 → 297); give it its
own `--out` folder. Neither marks a repeat as done; delete a smoke folder before the
full run. Override the interpreter with `SERVER_PYTHON=...` and add
vLLM flags with `EXTRA_ARGS="..."`. If the pinned weights are already in
`.cache/huggingface`, the run is offline (Llama is gated). Otherwise log in to
Hugging Face with approved access first.

A finished repeat has a `DONE` file and is skipped on rerun. An interrupted
repeat resumes in the same folder with a new server. That is fine if the device
passes the repeatability gate, but for a strict run delete the folder and rerun it.

Expected cost: about 100 tokens/s at batch size 1 on an RTX 3060 laptop GPU, i.e.
~45 min per repeat for all 1483 cases (GSM8K ~33 min, HumanEval ~10 min). An A100
or MI210 should be faster; P150b is unmeasured. Three repeats is a few hours per
device, so run a `--limit 5` smoke test first.

### Tenstorrent notes

- The P150b profile uses the same settings as the earlier TT run, including
  on-device sampling. That mode returns **no token IDs or log-probs**, so
  comparisons with P150b use text (first differing character) and have no margins.
- `--no-enable-prefix-caching` was not used in the earlier TT run. If the TT
  plugin rejects it, remove it from `common=(...)` in `run.sh` and note that in
  your results.
- This profile has not been run on P150b in this rework. Treat P150b as
  exploratory, as in Assignment 1.

## 2. Score on the evaluation host

Copy each device's `results/<device>/` folder into one place (for example the
RTX 3060 with bubblewrap), keeping the `<device>/r<k>/` layout:

```bash
.venv-client/bin/python normbench-rework/week1.py score normbench-rework/results/*/r*
```

Math is scored by numeric match. Code is run with HumanEval tests in bubblewrap,
with no unsandboxed fallback. Scoring is resumable.

## 3. Report

```bash
.venv-client/bin/python normbench-rework/week1.py report \
  --runs normbench-rework/results --out normbench-rework/report-v1
```

You can run this before scoring to see token-level results; accuracy columns
stay empty until then. Start with `summary.md`:

0. **Accuracy and drift.** Plain accuracy per device and workload, then for each
   device pair: *output drift* (% of answers whose text differs), *accuracy
   drift* (accuracy B − A in percentage points) and *flip rate* (% of questions
   whose correctness changed). These are the headline, DriftBench-style numbers.

1. **Gate.** Each device must show 100% repeat-vs-repeat identity. If not, find
   out why before reading cross-device numbers (this is how Qwen3.5 failed).
2. **Cross-device table.** Identity rate, cases that are stable on both devices
   but different (the real signal), the four outcomes, flip rate, and McNemar p.
3. **Margins.** The median top-1/top-2 log-prob gap at the first divergent token,
   compared with all tokens. A much smaller gap at divergences means flips happen
   at near-ties (RQ2).

Other outputs: `accuracy.csv`, `repeatability.csv`, `cross_device.csv`,
`divergences.csv` (one row per differing case: position, both tokens, both
margins, correctness), and `week2-cases.json`. The last file holds stable
divergent cases with their input IDs plus the shared generated prefix, ready
for Week 2 layer capture.

## Limits to state in the write-up

- Cross-device comparisons use repeat 1 on each side. `stable_on_both` says
  whether every repeat agrees, so restrict claims to stable cases.
- This compares complete stacks (hardware, driver, vLLM build, kernels), not
  hardware alone. Package versions are saved in each run's `meta.json`.
- Log-prob keys are token strings, so two candidates that decode to the same
  string would merge. This is rare and affects only the margin column.
- Requesting log-probs did not change the generated tokens in a local check
  (RTX 3060, 3 cases), but this was not verified on every device.

## How `data/cases.jsonl` was built

HumanEval comes from DriftBench's `humaneval_prompts.jsonl`; GSM8K is
`openai/gsm8k`, config `main`, split `test` (parquet sha256 `ee7b8da9…4f59`),
with the reference answer taken after `####`. `build_cases.py` renders each
prompt with the Llama chat template exactly as DriftBench does and tokenizes it
with the pinned tokenizer. It refuses to write unless all 664 prompts from the
earlier runs (164 code + 500 math) reproduce their saved token IDs exactly,
which they did.
