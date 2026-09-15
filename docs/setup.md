# Preparing the three inference hosts

Use the same repository revision on every machine. Keep the existing working
A100 serving environments; refactoring configs does not require reinstalling
packages. Runs capture the installed framework versions and hardware metadata.
A suite's `server_python` fields select the serving environments independently
of the benchmark client's Python.

## Common benchmark client and datasets

On a new inference host, this optional setup prepares a Python 3.12 tokenizer
client without CUDA/ROCm/torch packages and fetches the pinned prompt artifact:

```bash
bash scripts/bootstrap.sh --client-only
export DRIFTBENCH_PYTHON="$PWD/.venv-client/bin/python"
```

Use the same client/tokenizer environment across hardware when comparing drift.
`bash scripts/bootstrap.sh --datasets-only` only fetches the benchmark checkout.
Neither mode installs Bubblewrap or system packages. Copying completed bundles
to an evaluation host does not require fetching the dataset separately.

## A100 40 GB

Existing installations need no package changes. For a fresh NVIDIA installation:

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

Use `bash scripts/run_mi210.sh --run-id mi210-r01`. A client-only environment may
be selected with `DRIFTBENCH_PYTHON`; otherwise the wrapper uses ROCm vLLM's Python.
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

The wrapper loads that activation file. To drive it with the common client,
call `scripts/results.sh infer` directly after activation and after exporting
`DRIFTBENCH_PYTHON="$PWD/.venv-client/bin/python"`.

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
bash scripts/results.sh stage code results/runs/IMPORTED_SAFETY_RUN
bash scripts/results.sh stage final results/runs/IMPORTED_SAFETY_RUN
```

A failed isolation check remains an explicit error; generated code is never
executed on the host without the sandbox. Transfers and exact commands are in
[the staged workflow](staged-workflow.md).
