# Preparing the three inference hosts

Use the same repository revision on every machine. Keep the existing working
A100 serving environments; refactoring configs does not require reinstalling
packages. Runs capture the installed framework versions and hardware metadata.
A suite's `server_python` fields select the serving environments independently
of the benchmark client's Python.

## Common benchmark client and datasets

On every inference host, this required setup prepares a pinned Python 3.12.14 tokenizer
client without CUDA/ROCm/torch packages and fetches the pinned prompt artifact:

```bash
bash scripts/bootstrap.sh --client-only
```

Use the same client/tokenizer environment across hardware when comparing drift.
`bash scripts/bootstrap.sh --datasets-only` only fetches the benchmark checkout.
Neither mode installs Bubblewrap or system packages. A Transformers notice that Torch is absent in this client is expected; GPU checks run in the separate serving interpreter. Copying completed bundles
to an evaluation host does not require fetching the dataset separately.

## A100 40 GB

Prepare the common client even on existing installations. The doctor checks existing serving packages against `runtime-contracts.json`; restore the lockfile if they differ. For a fresh NVIDIA installation:

```bash
bash scripts/bootstrap.sh nvidia
bash scripts/install_sglang.sh
```

The pinned environments remain `.venv` (vLLM 0.17.1/CUDA 12.8) and `.venv-sglang`
(SGLang 0.5.10.post1/CUDA 12.8). The SGLang installer provides managed Python
headers and workspace GCC/CUDA compilers. The Qwen3.5 SGLang profile does not use
`--language-only`, which would require a separate encoder server.

Run `bash scripts/run_a100.sh --run-id a100-r01`. Scheduler device visibility is
preserved unless `--device` is supplied. Models run sequentially on one GPU;
this is not a multi-GPU tensor-parallel benchmark.

For safety evaluation on A100, use the existing NVIDIA environment with PyTorch,
Transformers and Accelerate, and approved access to Llama-Guard-3-8B:

```bash
export DRIFTBENCH_PYTHON="$PWD/.venv/bin/python"
bash scripts/results.sh stage safety results/runs/RUN_TO_SCORE
```

Use that same environment for safety classification of **all** inference hosts.
The guard runs after the serving process stops and releases its GPU memory.

## AMD MI210

Use ROCm builds of vLLM and SGLang prepared for MI210/gfx90a. CUDA lockfiles and
the NVIDIA SGLang installer must not be used for these serving environments.
The default paths in `suites/mi210.json` are:

```text
.venv-rocm-vllm/bin/python
.venv-rocm-sglang/bin/python
```

Install the runner into each existing vendor environment without replacing its
PyTorch/framework dependencies:

```bash
.tools/uv pip install --python .venv-rocm-vllm/bin/python --no-deps -e .
.tools/uv pip install --python .venv-rocm-sglang/bin/python --no-deps -e .
```

If your approved ROCm environments live elsewhere, change the two `server_python`
paths in the suite before beginning a run. Keep the serving versions and host
ROCm stack matched; model-specific Qwen3.5 hybrid-attention kernel support on
gfx90a still requires execution on MI210. This repository does not provide a
locally validated ROCm binary distribution. Consult the official
[vLLM ROCm installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
and [SGLang AMD guide](https://docs.sglang.io/docs/hardware-platforms/amd_gpu)
for the serving build; verify that the selected build targets gfx90a.

Use `bash scripts/run_mi210.sh --run-id mi210-r01`. The wrapper uses `.venv-client` for prompt preparation and orchestration, and the suite's ROCm interpreters for serving.
No judge or HumanEval runs here. Transfer complete inference to A100 for safety.

## Blackhole P150b

Only vLLM is in the active P150b suite. Follow [the TT installer guide](tt-vllm-install.md):

```bash
bash scripts/install_tt_vllm.sh
source .venv-tt-vllm/activate-tt.sh
bash scripts/run_blackhole_p150b.sh --run-id p150b-r01 --allow-experimental
```

This uses Python 3.12, vLLM 0.25.1, the pinned compatibility-tag TT plugin,
TTNN/TT-Metal 0.77.0 and SFPI 7.69.0. The built runtime comes from the TTNN wheel;
`TT_METAL_RUNTIME_ROOT` must not point at the unbuilt source checkout. The
installer's `--toolchain-only` repairs SFPI and regenerates activation without
reinstalling Python packages.

The wrapper loads the TT activation file for native runtime paths and selects the common tokenizer client. Older activation files that selected the TT serving Python are handled by the wrapper.

Use the chips assigned by your host provider through `TT_VISIBLE_DEVICES` and
the profile's `MESH_DEVICE=P150`; the runner does not reset or reconfigure boards.
The plugin implements device/model support, not this benchmark runner. Actual
P150b inference remains unverified here, and the earlier reported KMD/firmware
are below TT-Metal 0.77.0's documented baseline. See the TT guide for requirements.

## RTX machine: code and final evaluation

Reuse the existing RTX `.venv` and working Bubblewrap setup. No 7–9B inference
model or LlamaGuard weights need to fit in the RTX GPU. HumanEval executes in an
isolated CPU process; final math/long-context scoring uses saved responses.
Paired chat comparison uses a small embedding model on CPU.

```bash
export DRIFTBENCH_PYTHON="$PWD/.venv/bin/python"
bash scripts/results.sh stage evaluate results/runs/IMPORTED_SAFETY_RUN
```

A failed isolation check remains an explicit error; generated code is never
executed on the host without the sandbox. Transfers and exact commands are in
[the staged workflow](staged-workflow.md).

## Version policy

`requirements.client.lock.txt` pins every tokenizer-client dependency. Python and
all these package versions are checked before prompt preparation, regardless of
which serving framework is selected. The runner also stores a source hash so
comparisons expose different benchmark-client implementations.

| Serving stack | Framework | Torch release | Transformers |
|---|---|---|---|
| NVIDIA / AMD vLLM | 0.17.1 | 2.10.0 | 4.57.6 |
| NVIDIA / AMD SGLang | 0.5.10.post1 | 2.9.1 | 5.3.0 |
| TT vLLM | 0.25.1 | 2.11.0 CPU | 5.12.1 |

NVIDIA uses CUDA 12.8 builds; AMD requires corresponding ROCm builds, with actual
build suffixes/ROCm versions recorded. These AMD release targets are a comparison
contract, not a claim that a prebuilt MI210 wheel exists for every model. Use the
vendor build instructions and check with `doctor --suite suites/mi210.json`.
If a required vendor stack cannot meet a pin, record a deliberate change in
`runtime-contracts.json` and the suite, start new runs, and treat that comparison
as including software changes. Do not force incompatible packages with `--no-deps`.

The TT [compatibility branch](https://github.com/tenstorrent/vllm-tt-plugin/tree/compat/vllm-0.25.1)
targets vLLM 0.25.1. Its [NVIDIA dependency file](https://github.com/vllm-project/vllm/blob/v0.25.1/requirements/cuda.txt) includes Torch 2.11 and CUDA-13-specific dependencies, so it is not a drop-in upgrade for the existing R570/CUDA 12.8 environment. We retain the known CUDA 12.8 baseline rather than installing incompatible wheels to equalize a version number. Matching package version strings alone would
also not equate CUDA, HIP and TT kernels or TT internal precision. Comparisons
retain these differences instead of claiming hardware-only isolation.

## CUDA detection on A100

Driver 570.195.03 is in the CUDA 12 driver family. NVIDIA documents CUDA 13 as
requiring R580 or newer under its standard compatibility rules; installing a
CUDA compiler does not replace the CUDA runtime embedded in Torch wheels.
[NVIDIA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Inside the allocated GPU job/container, run:

```bash
bash scripts/results.sh doctor --suite suites/a100.json --framework vllm sglang \
  --output results/a100-doctor.json
```

This uses the actual suite interpreters, tests CUDA allocation/arithmetic, and
checks package and CUDA-runtime pins. A failed initialization gets one fresh-process
retry; each probe times out after 30 seconds. The serving launcher writes the
same information to `settings/SETTING/startup-diagnostics.json` before loading
weights. Passing this probe is not a guarantee that every model kernel will work.

If detection fails:

- Check the reported device visibility. Preserve Slurm's `CUDA_VISIBLE_DEVICES`;
  do not set `--device 0` to guess a physical GPU on a managed allocation.
- If CUDA 13 packages slipped in, restore the CUDA 12.8 lockfile using the NVIDIA
  installer. Do not upgrade the node driver from this repository.
- Remove CUDA `stubs` directories from runtime `LD_LIBRARY_PATH` when diagnosed.
- If the CUDA 12.8 calculation still fails, request a healthy allocation or show
  the administrator the diagnostic report and `nvidia-smi` output. Earlier `ERR!`
  device readings are evidence to investigate with the host administrator, not
  proof that a Python reinstall can repair the node.

The runner never resets GPUs, changes Slurm allocations or retries an entire
inference workload automatically after a serving failure.
