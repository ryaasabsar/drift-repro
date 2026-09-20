# Locate a divergence and export a kernel microbenchmark

This workflow captures tensors **inside the installed vLLM worker**, compares
identical math/code prefixes, and exports actual operation operands for replay
against a CPU float64 reference. It does not change your benchmark suites or
install packages. Use the existing NVIDIA or ROCm serving environment.

## 1. Capture layer boundaries

Run from the repository root inside an allocated GPU job:

```bash
bash scripts/investigate.sh capture-serving \
  --bundle results/investigations/mi210-a100/cases-v1 \
  --requests results/investigations/mi210-a100/report-v1/probe-cases.json \
  --model qwen35_08b --condition serial \
  --cases humaneval_146 gsm8k_256 \
  --device a100 --server-python .venv/bin/python \
  --processes 2 --repeats 2 \
  --output results/investigations/causal/a100-boundaries
```

For MI210 change `--device mi210`, `--server-python .venv-rocm-vllm/bin/python`,
and the output name. For the laptop use `--device rtx3060`. Copy the entire
frozen bundle and probe request file to each host. `--dry-run` verifies the
prefixes and prints the configuration without loading a model. Existing
outputs are never overwritten; use another name after a failed/completed run.

Two fresh worker processes each capture both prefixes twice. Uninstrumented
requests before and after each pair check whether instrumentation changes the
next token or top-k scores. Parameter hashes before and after check weight
stability. Captures include embeddings, block boundaries, normalization outputs,
and raw vocabulary logits. There is no safety judge or generated-code execution.

These are explicit diagnostic settings: serial, model-level eager, 2048-token
context, fixed 512 MiB KV cache, and one output token. Prefixes are never
truncated. Keep `--context` and `--kv-cache-mib` equal across machines if you
change them. Chunked prefill remains enabled for compatibility; each prefix
fits the configured prefill budget. These are not benchmark accuracy runs.

**Eager caveat:** vLLM 0.17.1's `GemmaRMSNorm.forward_cuda` internally calls
`torch.compile`. Model-level eager mode does not rule out every compiled kernel.

## 2. Compare within a process, across restarts, then across machines

```bash
bash scripts/investigate.sh compare-serving \
  results/investigations/causal/a100-boundaries/process-001/case-000/capture-001 \
  results/investigations/causal/a100-boundaries/process-001/case-000/capture-002 \
  --output results/investigations/causal/within-process-code
```

Next compare `process-001` with `process-002`, using `capture-001` for the same
case. Then compare A100 with MI210. In this example, `case-000` is code and
`case-001` math; `plan.json` records the exact order.

Read `comparison.json` and `tensors.csv`. `first_different_output` locates the
first observed differing boundary. Inspect its **inputs** too: if they already
differ, the cause is upstream. Review `weights_equal`, `weights_unchanged`, and
`observer_check_equal` in the comparison and each process's `capture-run.json`.
A failed observer check means this trace may not represent the original failure.
The check covers tokens/top-k, not hidden values in an uninstrumented run.

## 3. Narrow to the operation

Repeat the capture with `--module` set to an exact name from `modules.json`,
and a new output directory. Examples for Qwen:

```text
--module language_model.model.layers.0
--module language_model.model.layers.0.linear_attn.chunk_gated_delta_rule
--module language_model.model.layers.0.linear_attn.norm
```

Choose the observed suspect, not layer 0 by assumption. A layer target traces
descendants and registered ATen/custom operations. The gated-delta adapter
captures q, k, v, gates, beta, **initial recurrent state**, sequence boundaries,
and both outputs. The gated-normalization adapter includes learned parameters.
Compare targeted traces again and inspect `operation_candidates`.
`inputs_identical=true` plus a changed output is a useful isolated replay
candidate. It is not by itself proof of a hardware fault.

The default limit is 512 MiB per request. A budget overrun fails explicitly;
narrow the module before increasing `--max-mib`. Host copies and tracing are
intrusive, so do not use capture timings as inference performance results.

## 4. Export actual operands and replay on both GPUs

Replace `EVENT_ID` below with the integer from the targeted trace/comparison:

```bash
bash scripts/investigate.sh export-operation \
  --trace results/investigations/causal/a100-target/process-001/case-001/capture-001 \
  --event EVENT_ID \
  --output results/investigations/causal/operation-case

DRIFTBENCH_PYTHON=.venv/bin/python bash scripts/investigate.sh replay-operation \
  --bundle results/investigations/causal/operation-case \
  --device cuda --repeats 3 --profile \
  --output results/investigations/causal/replay-a100
```

Copy the **same complete operation-case directory** to MI210. Replay using
`DRIFTBENCH_PYTHON=.venv-rocm-vllm/bin/python` and output `replay-mi210`.
No checkpoint download, original capture directory, or serving model is needed.
PyTorch uses the device name `cuda` for both CUDA and ROCm.

```bash
bash scripts/investigate.sh compare-operation-replays \
  results/investigations/causal/replay-a100 \
  results/investigations/causal/replay-mi210 \
  --output results/investigations/causal/operation-comparison.json
```

`replay.json` reports error against float64, error/bitwise agreement against the
captured output, repeatability, hardware/software, and tolerances. `outputs.npz`
preserves actual output bits and references. `kernel-trace.json` optionally
records real profiler events so you can inspect which physical kernels ran.
There is no automatic timing claim.

Default tolerances are zero. Add `--atol`/`--rtol` only with a justified numerical
acceptance criterion. BF16 disagreement with float64 is not automatically a bug:
reduction order, fusion, approximate functions and intermediate rounding matter.
The reference starts from the exact captured, already-quantized inputs; it is
higher precision, not an exact-real-arithmetic oracle. The gated-delta reference
uses a sequential recurrence, independently of the production chunked algorithm.

Supported adapters include dense matrix products, selected reductions and
elementwise operations, vLLM fused RMS/SwiGLU, gated RMS normalization, and the
FLA chunked gated-delta path used by these A100/MI210/RTX profiles. Other opaque,
stateful or aliased operations fail export rather than being substituted with
an unrelated operation. FlashInfer GDN on other architectures is not covered.

## Files to move

```text
a100-boundaries/
  campaign.json
  process-001/
    plan.json, worker.log, modules.json, capture-run.json
    weights-before.json, weights-after.json
    case-000/capture-001/{trace.json,tensors.npz}
    case-000/capture-002/{trace.json,tensors.npz}
    case-001/...
  process-002/...
operation-case/
  operation.json, tensors.npz, README.md
replay-a100/
  replay.json, outputs.npz, kernel-trace.json
```

Move a whole campaign for provenance, or just the operation bundle for replay.
Do not move virtualenvs, caches, or credentials. Numeric NPZ files load without
pickling. Hashes reject modified or incomplete artifacts, and comparisons
require matching model/prefix/settings or matching operand-bundle fingerprints.

To establish a cause, make a targeted intervention at the isolated operation,
repeat the check, and confirm its effect on the original math/code result.
Locating a differing tensor alone does not establish a particular rounding rule.
