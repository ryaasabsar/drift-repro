# NormBench

A standalone numerical benchmark for **gated RMSNorm and its primitive
operations** on NVIDIA, AMD, and Tenstorrent accelerators. It compares identical
inputs, FP32/BF16 intermediates, raw output bits, and errors against frozen
float64 references. No model, inference server, credentials, or DriftBench
installation is required.

For a short GPU example, run `python simple_divergence.py` in your accelerator
environment. It computes sigmoid and BF16 rounding for the input at `(761, 11)`.
The reduced kernel may compile differently; `python reproduce_examples.py`
replays the original full-shape kernels and prints the three recorded examples.

Copy this entire directory into its own repository or onto another machine.
The Python package, launcher, tests, and approximately 20 MB of numeric fixtures
are all included here. Results and compiler caches stay outside version control.

For the next A100–MI210 experiment, use the [sigmoid/rsqrt intervention guide](INTERVENTIONS.md).
It provides four measured variants, unchanged-value controls, and compiled-kernel exports.

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

## Matching the A100 and MI210 software

From the parent repository root, inside an allocated GPU job:

```bash
# A100 (also usable on the RTX 3060)
bash normbench/install_matched.sh nvidia
normbench/.venv-matched-nvidia/bin/python normbench/simple_divergence.py

# MI210
bash normbench/install_matched.sh amd
normbench/.venv-matched-amd/bin/python normbench/simple_divergence.py
```

Both dedicated environments pin Python **3.12.14**, PyTorch **2.9.1**, Triton
**3.5.1**, NumPy **2.2.6**, and identical versions/hashes of the common Python
dependencies. NVIDIA uses the CUDA 12.8 wheel; AMD uses the ROCm 6.4 wheel and
`pytorch-triton-rocm==3.5.1` (which supplies the `triton` module). The two lockfiles
pin platform wheels and transitive dependencies with SHA256 hashes. They follow
the [official PyTorch 2.9.1 builds](https://pytorch.org/get-started/previous-versions/#v291).

The installer creates `.venv-matched-nvidia` or `.venv-matched-amd` inside this
directory. Existing serving environments and node drivers are retained. There
is no vLLM dependency for this experiment. `--dry-run` prints the plan; `--check`
rechecks an existing installation. Installation checks package versions, a GPU
calculation, and compilation/execution of the small Triton example. It saves
`environment.json` (including the PyTorch commit and compiler) and `installed.txt`
inside the environment. Preserve these files alongside transferred results.

CUDA 12.8 fits the reported A100 driver 570.195.03. ROCm 6.4 is the userspace
build offered for official PyTorch 2.9.1; it does not replace `/opt/rocm` on the
MI210 node. AMD documents [driver/userspace compatibility](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.0/reference/user-kernel-space-compat-matrix.html).
The reported HIP 7.0 version alone does not identify the node's kernel driver;
the GPU checks must pass on the actual allocation. No sudo is required.

For the numerical comparison, use the same repository revision and run:

```bash
# A100
NORMBENCH_PYTHON="$PWD/normbench/.venv-matched-nvidia/bin/python" \
  bash normbench/run.sh a100 --cases captured bf16-midpoints --precisions float32 \
  --output normbench/results/a100-matched-v1

# MI210
NORMBENCH_PYTHON="$PWD/normbench/.venv-matched-amd/bin/python" \
  bash normbench/run.sh mi210 --cases captured bf16-midpoints --precisions float32 \
  --output normbench/results/mi210-matched-v1

# After transferring both complete result folders to one machine (NumPy only):
bash normbench/run.sh compare normbench/results/a100-matched-v1 \
  normbench/results/mi210-matched-v1 --output normbench/results/a100-mi210-matched-v1
```

Run the same commands in the original environments with distinct output folders
to obtain software-before/after comparisons using identical benchmark source.
Changing PyTorch, Triton, and ROCm userspace together tests the software stack;
it does not isolate Triton's version alone. Matching version numbers still
leaves different compiler backends, device libraries, drivers, and GPU hardware.
Record disagreements even when versions match.

Start with these ordinary benchmark runs. The existing intervention bundle
requires reproducing the **old** baseline; a changed stack may correctly fail
that control. If new matched baselines pass their controls, use
`prepare-intervention` to freeze a new shared bundle before rerunning interventions.
Do not bypass a failed historical-baseline check. The Tenstorrent TTNN stack is
separate and is not modified by this A100/MI210 installer.

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
