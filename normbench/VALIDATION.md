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
