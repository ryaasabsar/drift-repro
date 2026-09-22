# Standalone validation — 2026-09-20

The project was copied to an unrelated temporary directory. Its source checkout
was absent from `PYTHONPATH`. Default CPU and NVIDIA workers ran with import
hooks rejecting both `driftbench_runner` and `vllm`.

- **16 tests passed**, including custom numeric input preparation, integrity
  checks, injected numerical differences, and a CLI dry run blocking all
  accelerator/framework imports.
- **RTX 3060:** all six cases, FP32/BF16 stages, three calls in each of two fresh
  processes. All 281 recorded stage outputs were repeatable within and across
  processes. Instrumented outputs matched the standalone fused control for
  every case/repetition. BF16 midpoint casts matched the rounded reference.
- **CPU:** all six cases, both precisions, three calls and two processes. All
  221 recorded outputs were repeatable. No fused GPU control is claimed.
- **Optional vLLM control:** the captured case and midpoint probes passed on
  the RTX 3060, with FP32 stages, two calls and two processes. Instrumentation
  matched the installed vLLM control before/after and for every repetition.
- **Packaging:** built and installed the wheel offline without dependencies.
  Its module loaded the bundled fixture from an unrelated working directory.
  That installed package compared the CPU/RTX campaigns using NumPy only,
  producing 442 paired stage rows and explicitly reporting unavailable controls.

Runtime used: Python 3.12, PyTorch 2.10.0+cu128, Triton 3.6.0, NumPy 2.2.6.
The fixture fingerprint was
`c145dd31a7f6aca01c01a0fbc976cb4b72fef07be570e25d1459a85be3e0aff8`.

Local artifacts are retained under `results/validation-20260920/` (ignored by
Git). Plans record the temporary execution paths for provenance; comparisons
work after moving complete result folders.

A100, AMD MI210, and Tenstorrent runtime validation remains to be performed on
those machines. Local repeatability and matching instrumentation do not prove
cross-device bitwise agreement or identical generated instructions.

## Intervention experiment — 2026-09-22

The sigmoid/rsqrt experiment has 29 passing tests, including rejection of
different source operands, failed source observer checks, changed bundles,
changed compiler artifacts, incompatible protocols, and nonrepeatable results.
Synthetic comparisons verify that resolved and newly introduced differences
are counted independently and that failed controls invalidate attribution even
when output differences disappear.

The final local RTX 3060 campaign is
`results/interventions-rtx3060-validation-v3/`: 96 stage/output records per
process, three repetitions, and two fresh processes. All recorded tensors
repeat within and across processes. Each worker exports 16 compiled kernels
(eight paths with/without diagnostic stores), including IR, PTX and cubin.
The original kernel and replay bridge reproduce the frozen NVIDIA baseline.

The controls deliberately do **not** all pass. `self_rsqrt` and `self_both`
change 39,424 `normalized` diagnostic values (maximum 4 FP32 ULP), while their
native BF16 outputs and other snapshots stay unchanged. This occurs despite
matching compiled binary hashes. The control differences are retained in
`control-differences.csv`; shared-rsqrt/shared-both attribution remains false.
Original/shared-sigmoid controls pass. This is local diagnostic evidence, not
an A100/MI210 intervention outcome or proof of a specific compiler transformation.

The original CPU CLI also completed a fresh-process regression campaign after
the shared process runner was extracted. AMD export interfaces were checked
against Triton 3.4.0 source and tested with fixture objects; actual MI210
intervention execution remains to be done. A100 intervention execution likewise
remains to be done on the user's node.
