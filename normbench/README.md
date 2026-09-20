# NormBench

A standalone numerical benchmark for **gated RMSNorm and its primitive
operations** on NVIDIA, AMD, and Tenstorrent accelerators. It compares identical
inputs, FP32/BF16 intermediates, raw output bits, and errors against frozen
float64 references. No model, inference server, credentials, or DriftBench
installation is required.

Copy this entire directory into its own repository or onto another machine.
The Python package, launcher, tests, and approximately 20 MB of numeric fixtures
are all included here. Results and compiler caches stay outside version control.

## Environment

Activate an existing environment for the hardware you will use:

| Command/backend | Required packages |
|---|---|
| Prepare, compare, dry run | NumPy |
| CPU test harness | NumPy and PyTorch |
| NVIDIA | NumPy, CUDA PyTorch, and Triton |
| AMD | NumPy, ROCm PyTorch, and the matching Triton runtime |
| Tenstorrent | NumPy, PyTorch, and TTNN in an initialized TT-Metal environment |

The launcher uses the active `python3`. It installs nothing and does not source
another project's environment. Alternatively set
`NORMBENCH_PYTHON=/absolute/path/to/python`. GPU libraries and drivers must match
the machine; the package deliberately does not install a CUDA build of PyTorch
onto an AMD or Tenstorrent host. Python 3.12 is supported (minimum: 3.10).

Installing NormBench is optional when using `run.sh`. For a Python package or
console command, run `python -m pip install .` from this directory in the
selected environment. Only NumPy is a mandatory package dependency.

## Run on each machine

From this directory, inside your allocated accelerator job:

```bash
# NVIDIA A100
bash run.sh a100 --output results/a100-v1

# AMD MI210
bash run.sh mi210 --output results/mi210-v1

# Tenstorrent Blackhole P150b
bash run.sh p150b --output results/p150b-v1
```

The aliases `nvidia`, `amd`, `tenstorrent`, and `rtx3060` also work. The launcher
preserves scheduler visibility masks. NVIDIA/AMD use the first visible GPU;
Tenstorrent uses `--tt-device-id 0` by default. Set the TT device ID to the device
assigned to your job, and initialize your TT runtime environment beforehand.

A dry run checks the fixture and prints the plan without accelerator imports:

```bash
bash run.sh p150b --dry-run
```

Defaults are **two fresh processes, three repetitions, all six cases, both
FP32 and BF16 staged paths**. Keep the code, fixtures, and options identical
between machines. For a shorter diagnostic, add
`--cases captured bf16-midpoints --precisions float32`. Existing results cannot
be overwritten; choose a new output folder for each run.

The installed package also supports
`python -m normbench run --backend nvidia --output results/a100-v1` and the
`normbench` console command. `cpu` is a test harness without a fused GPU control.

An optional `--control vllm` on NVIDIA/AMD uses an already installed vLLM
`rmsnorm_fn` as the native control. The default does **not import vLLM**. This
optional adapter targets the vLLM 0.17.1 internal API and records its source hash;
other versions may require adapter changes. Use the same control option on
both hosts. Results from the earlier embedded implementation have a different
code/protocol fingerprint; generate fresh campaigns with this standalone code.

## What runs

| Path | NVIDIA / AMD | Tenstorrent |
|---|---|---|
| `native` control | Bundled fused Triton kernel, BF16 inputs/output | Native TTNN FP32 RMSNorm, separate accurate SiLU gate, BF16 cast |
| `instrumented` | Diagnostic Triton kernel, FP32 intermediate snapshots; checked against native output before/after | Explicitly unavailable; no Triton emulation |
| `pipeline` | PyTorch eager operations, each stage materialized at the selected precision | TTNN operations, each stage materialized at the selected precision |
| `isolated` | Each PyTorch primitive gets frozen identical operands | Each TTNN primitive gets those same frozen operands |

Tenstorrent's native control is a **TTNN composition**, not a claim that it is
the fused implementation in vLLM-TT. The TTNN path targets the TTNN 0.77.0
API, using HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True` for
reduction/RMSNorm, and accurate reciprocal-square-root/sigmoid calls. Other
operators use their TTNN defaults, which are recorded through package versions.
There is no host-computation fallback or automatic BF16 fallback for FP32
operations. An unsupported operation/dtype fails with a worker log.

The fused Triton kernel follows the inspected vLLM 0.17.1 normalization
formula and launch geometry, implemented locally. Its instrumented variant
adds FP32 intermediate snapshots. Additional stores can change compilation. If
`instrumented_matches_native` or `native_before_after_equal` is false, its
intermediates cannot be attributed to the selected control kernel. Even matching
outputs do not prove every generated instruction is identical. Staged PyTorch
or TTNN paths are deliberate interventions, not invisible instrumentation.

The stages are:

1. `square`: elementwise x squared.
2. `sum_squares`: reduction across the row width.
3. `mean_square`: divide by width.
4. `add_epsilon`: add epsilon.
5. `rsqrt`: reciprocal square root.
6. `normalized`: multiply x by inverse RMS.
7. `weighted`: multiply by learned weights.
8. `sigmoid`: sigmoid of gate input z (an independent branch).
9. `gate`: z times sigmoid(z).
10. `precast`: weighted normalization times gate.
11. `output`: device conversion to BF16.

The BF16 staged path intentionally rounds between operations; the FP32 path
retains FP32 intermediates until the final BF16 output. Both begin with the
same captured BF16 input values. Isolated stages use frozen rounded reference
operands, so differences cannot propagate from earlier device calculations.
For those isolated stages, reference errors are computed from those exact
operands. Native/instrumented/pipeline errors use the complete float64 formula.

## Inputs and reference

The bundled [fixtures](normbench/fixtures/gated_norm) contain:

- `captured`: 1552×128 numeric tensors captured from a Qwen3.5-0.8B normalization
  operation, preserving its launch shape. No checkpoint or original run folder
  is required. Focus positions: `(761,11)`, `(1330,36)`, `(1385,8)`.
- `synthetic-scale-0.5`, `synthetic-scale-1`, `synthetic-scale-2`: shared seeded
  BF16 inputs with only the x scale changed; gate values cover both signs.
- `epsilon-dominated`: zero and tiny rows.
- `bf16-midpoints`: FP32 values immediately below, at, and above BF16 midpoints
  near 0.5, 1, 2 and 4, both signs and signed zero. These inputs remain FP32
  until the device cast, even in a BF16-staged campaign.

The manifest records provenance and hashes. Float64 references are frozen once,
avoiding changes to the reference on each host. They are higher-precision
numerical references, not exact real arithmetic. For ULP distances, float64 is
rounded **directly** to the target representation, ties to even; first converting
to FP32 could double-round near the BF16 boundaries under investigation.
Nonfinite results are counted separately.

Ordinary runs use the bundled fixture. To supply different normalization inputs,
create a numeric NPZ containing exactly `x`, `w`, and `z`: `x` and `z` have shape
`[rows, columns]`, and `w` has shape `[columns]` or `[1, columns]`. Values must be
finite and already BF16-representable, stored as FP32. Then freeze references:

```bash
bash run.sh prepare --inputs tensors.npz --epsilon 1e-6 --output fixtures/custom
bash run.sh a100 --fixture fixtures/custom --output results/custom-a100-v1
```

Omitting `--inputs` regenerates references from the bundled captured tensors.
Generate a fixture once and distribute the complete directory to all hosts;
do not regenerate it independently on each machine.

## Collect and compare

Copy the **complete** `a100-v1`, `mi210-v1`, and `p150b-v1` output folders onto
one analysis machine. They contain numeric results and metadata, not models or
environments. The comparison needs only NumPy and does not open an accelerator.

```bash
bash run.sh compare results/a100-v1 results/mi210-v1 --output results/a100-mi210-v1
bash run.sh compare results/a100-v1 results/p150b-v1 --output results/a100-p150b-v1
```

| Artifact | Contents |
|---|---|
| `campaign.json` | Completion/failure and process coverage |
| `process-*/worker.log` | Backend errors and progress |
| `process-*/result.json` | Environment, controls, per-stage hashes, repeats and errors |
| `process-*/reference-errors.csv` | Error against float64, rounded-reference ULP distances |
| `process-*/focus-values.csv` | Every stage's values at the three focus positions |
| `process-*/outputs.npz` | Raw logical FP32/BF16 output bits; no pickle |
| `restart-*.json/csv` | First process versus each fresh process |
| Comparison `README.md`, `stages.csv` | First observed differing stages and complete cross-device metrics |
| Comparison `mismatch-examples.csv` | Up to five differing coordinates/values/raw bits per stage, when present |

Start with repeatability, then instrumentation controls, then isolated primitive
differences. A pipeline mismatch can have an upstream cause. An isolated
mismatch occurs with identical operands, but may reflect different compiler,
library, or hardware choices. PyTorch and TTNN primitives need not use the same
algorithm. The sigmoid branch is independent of the reduction branch; a single
"first stage" is a reporting order, not proof of a causal chain between branches.

No pass/fail threshold is inferred from bitwise disagreement. This benchmark
does not measure throughput or workload accuracy. Once a cause is isolated,
test a controlled change and then return to the model to measure its effect.

## Tests and API provenance

See [local validation](VALIDATION.md) for detached CPU/RTX runs, packaging checks,
and the limits of hardware coverage.

Run `python -m pytest` from this directory after installing the test extra with
`python -m pip install '.[test]'`. CPU execution tests also require PyTorch.
Remote accelerator runtime validation must be done on the corresponding device;
unit tests and dry runs alone do not establish hardware compatibility.

The TTNN adapter was checked against the 0.77.0 API at TT-Metal commit
`9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`:
[unary bindings](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp),
[reduction bindings](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp),
[normalization bindings](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp).
