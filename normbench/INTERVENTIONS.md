# A100–MI210 causal intervention experiment

This experiment tests whether the observed sigmoid and reciprocal-square-root
(`rsqrt`) differences contribute to the three captured BF16 output differences.
It evaluates the entire 1552 × 128 output, including newly introduced differences.
It does not load a model or change the installed accelerator packages.

## Run on each machine

Pull the same source revision on both machines. The frozen intervention bundle
is included in `normbench/fixtures/a100-mi210-intervention/`: neither machine
needs the previous result folders. Keep your current accelerator environment
for the first experiment.

From the parent repository root, inside an allocated GPU job:

```bash
# A100
NORMBENCH_PYTHON="$PWD/.venv/bin/python" \
  bash normbench/run.sh intervene --backend nvidia \
  --output normbench/results/a100-interventions-v1

# MI210
NORMBENCH_PYTHON="$PWD/.venv-rocm-vllm/bin/python" \
  bash normbench/run.sh intervene --backend amd \
  --output normbench/results/mi210-interventions-v1
```

For a standalone checkout, activate its CUDA/ROCm Python environment, then run
`bash run.sh intervene --backend nvidia --output results/a100-interventions-v1`
or use `--backend amd`. An explicit `NORMBENCH_PYTHON` takes precedence over the
active environment. Scheduler visibility masks are preserved.

Defaults are three repetitions and two fresh processes. `--dry-run` verifies the
bundle and prints the plan using only NumPy. Existing output folders cannot be
overwritten; use a new name when retrying. No package upgrade or model download
is needed. Compilation requires the same working host compiler/Triton setup as
the original normalization benchmark.

Copy the **complete** two output folders onto one analysis machine, then run:

```bash
NORMBENCH_PYTHON="$PWD/.venv-client/bin/python" \
  bash normbench/run.sh compare-interventions \
  normbench/results/a100-interventions-v1 \
  normbench/results/mi210-interventions-v1 \
  --output normbench/results/a100-mi210-interventions-v1
```

Comparison needs NumPy only. Begin with the generated `README.md` and
`effects.csv`. Keep the original A100/MI210 results as separate baseline evidence.

## Four measured variants and their controls

| Variant | Operation output substituted |
|---|---|
| `original` | None; calls the unchanged original Triton JIT kernel |
| `shared_sigmoid` | Sigmoid only |
| `shared_rsqrt` | Reciprocal square root only |
| `shared_both` | Both |

Shared values are computed once in float64 and rounded once to FP32. The rsqrt
operand is the **actual FP32 `add_epsilon` tensor** that agreed between the
source A100 and MI210 traces. It is not reconstructed from an idealized reduction.
Sigmoid uses the exact frozen gate inputs. This preserves operand provenance.
The bundled references are numerical float64 references, not exact-real oracles.

The diagnostic kernel computes the original operations and loads the shared
values; device-resident switches select which values to consume. Its original,
shared, and self-value arms execute identical compiled code for each capture
mode. It runs these additional controls automatically:

- `replay_original`: the diagnostic binary with both switches off. Its outputs
  and snapshots are compared with the original kernel.
- `self_sigmoid`, `self_rsqrt`, `self_both`: inject that device's own diagnostic
  operation outputs. Reproducing these values must preserve its result.
- Every variant runs both with and without diagnostic stores. Final outputs
  must agree before interpreting snapshots.
- Uploads preserve input bits; unaffected branches, historical baseline
  reproduction, within-process repetitions, and fresh processes are checked.
- Compiled binary hashes must agree between the shared and corresponding
  self/original replay arms. Kernel and exported-source files are checked when
  comparing transferred results.

A false control is useful evidence of compilation/observer effects. It must
not be hidden by dropping difficult coordinates, changing the output dtype,
accepting only final-output equality, or declaring a successful causal result.
A completed campaign may have false controls; read the comparison eligibility.

## Reading the result

`effects.csv` has one row per primary variant per fresh process:

- `baseline_differing`: original A100/MI210 differing BF16 elements (expected 3
  if both reproduce the source observation).
- `resolved`: original differing coordinates that now agree.
- `remaining`: original differing coordinates that still differ.
- `new`: previously agreeing coordinates that now differ.
- `differing`, `differing_percent`: all current differences, with the full
  198,656-element denominator.
- `attribution_eligible`: all relevant controls, including restarts, passed.
- Reference mismatch percentage and relative L2 error are provided separately
  for each side; fewer cross-device differences need not mean better accuracy.

The working hypotheses are that shared sigmoid removes the differences at
`(761,11)` and `(1330,36)`, shared rsqrt affects `(1385,8)`, and sharing both
removes all three. These are predictions to test. Any variant may leave old
mismatches, introduce new ones, or fail its controls.

`coordinates.csv` includes every original or newly differing output coordinate.
`stages.csv` compares all saved native outputs and intermediate snapshots.
Worker `focus-values.csv` tracks all stages at the three original coordinates.
`controls.json` explains failed gates; `control-differences.csv` identifies the
affected intermediate stages and example coordinates when replay checks fail.
`comparison.json` includes both devices'
controls and package/runtime versions.

## Inspect compiled code after the controlled result

Each worker includes `kernels/<variant>-native/` and `*-trace/` with:

- Triton IR, GPU IR, LLVM IR, compiler metadata and launch geometry.
- NVIDIA: PTX and the actual compiled cubin. PTX is a virtual ISA; cubin is
  retained for optional machine-code disassembly with NVIDIA tools.
- AMD: AMDGCN assembly and the compiled HSACO.
- File hashes and instruction excerpts matching rsqrt/sqrt, exponential,
  reciprocal/division, fused multiply-add, and conversion terms.

`source/` preserves the exact Python modules used in the run. Missing required
compiler artifacts fail the run explicitly. If binary-only caching was enabled,
disable it and use a fresh `TRITON_CACHE_DIR` for the retry.

Start by examining the original native kernel and the replay bridge, then the
shared/self diagnostic binary. An instruction search match is not automatic
proof of the root cause. Passing intervention controls supports a contribution
from the substituted operation; the compiled code helps distinguish precision,
approximation, and compiler transformations. The existing PyTorch/Triton versions
differ between A100 and MI210, so do not attribute the finding to hardware alone.

After identifying a specific implementation difference, test a targeted kernel
change with fixed operands. Then return to the original math/code examples to
measure logit, token, and task-score effects. Shared reference loads are a
causal diagnostic, not a proposed inference optimization.

## Regenerating a bundle for another verified baseline

This is optional; the included bundle is already prepared from the supplied
A100/MI210 results. On a machine holding both complete baseline folders:

```bash
bash run.sh prepare-intervention results/a100 results/mi210 \
  --output fixtures/new-intervention
bash run.sh intervene --backend nvidia --bundle fixtures/new-intervention \
  --output results/new-a100
```

Distribute the exact same complete bundle to both machines. Preparation rejects
changed archives, failed observer controls, unstable source runs, different
protocols, or different actual rsqrt operands. Original normalization fixtures
and previous result files are retained unchanged.

Compiler export uses the Triton compiled-kernel artifact interface available in
[3.4.0](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/compiler/compiler.py)
and checked locally with 3.6.0. AMD assembly/binary stages are defined in the
[AMD backend](https://github.com/triton-lang/triton/blob/v3.4.0/third_party/amd/backend/compiler.py).
