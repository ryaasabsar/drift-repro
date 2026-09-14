# Qwen2.5-7B across serving frameworks

The new presets keep the same Qwen2.5-7B-Instruct revision, prompt formatting, seed,
generation settings, BF16 dtype, 32,768-token context and client batch size as the
existing A100/MI210 vLLM presets. All five workloads are selected (2,284 responses
per framework). Safety evaluation stays on your fixed A100 Llama Guard environment;
code execution stays on your RTX system. The [staged workflow](staged-workflow.md)
applies to these suites unchanged.

## Environments and support

| Host | Framework | Suite serving Python | Reference version |
|---|---|---|---|
| A100 | vLLM | `.venv/bin/python` | 0.17.1 |
| A100 | SGLang | `.venv-sglang/bin/python` | 0.5.10.post1 |
| A100 | TensorRT-LLM | `.venv-trt/bin/python` | 1.2.1 |
| MI210 | vLLM | `.venv-rocm-vllm/bin/python` | ROCm-compatible build |
| MI210 | SGLang | `.venv-rocm-sglang/bin/python` | ROCm-compatible build; 0.5.10.post1 CLI reference |

TensorRT-LLM is NVIDIA-only. NVIDIA lists A100 in its
[v1.2.1 hardware support](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.2.1/docs/source/supported-hardware.md).
The installed TensorRT-LLM PyTorch backend includes `Qwen2ForCausalLM`.
SGLang has a [ROCm integration](https://docs.sglang.io/docs/hardware-platforms/amd_gpu),
but MI210/gfx90a support must be verified for the actual build. These new profiles are
marked `hardware_pending`: local configuration validation is not physical validation.

Install the NVIDIA environments from the project root:

```bash
bash scripts/install_sglang.sh
bash scripts/install_tensorrt.sh
```

These scripts create the suite's `.venv-sglang` and `.venv-trt` environments with
Python 3.12, restore the pinned dependencies, install the runner, and prepare workspace
GCC 13 and CUDA compiler files (12.8.1 for SGLang, 13.0.2 for TensorRT-LLM). They need
Linux x86_64, `python3`, `curl`, network access and writable workspace storage. They
require no `sudo`, system package installation or bubblewrap. The existing `.venv`
client/vLLM environment is preserved. Rerunning an installer repairs its own environment
from the snapshot, so keep custom package changes in a separate environment.

Use `--dry-run` to inspect the commands without downloading or changing files, or
`--check` to verify an existing installation. Checks cover dependency consistency,
compiler execution, the PyTorch CUDA build and serving CLI imports. They do not load a
model or establish inference support on A100. Installation output can be saved with
`bash scripts/install_sglang.sh 2>&1 | tee sglang-install.log` (use `set -o pipefail`
if scripting this command). Large framework wheels and their download cache require
substantial free disk space; these local environments occupy about 11 GB and 15 GB,
excluding models, toolchains and caches.

The CUDA 13 TensorRT environment needs a compatible host NVIDIA driver. The scripts
do not install or upgrade the driver; a compiler download cannot supply driver support.
If installation succeeds but the GPU smoke test fails, inspect that framework's server
log before starting a full run.

For MI210, prepare ROCm environments using the framework's installation instructions;
do not apply the NVIDIA lockfiles there. The SGLang profile selects Triton attention and
disables AITER. Both model and kernel support still need checking on the actual MI210.
Install the runner in each serving environment with `python -m pip install -e . --no-deps`,
and ensure its client dependencies are available without replacing the framework's PyTorch.
`DRIFTBENCH_PYTHON` selects the client/evaluation Python; it does **not** replace the
per-framework `server_python` entries in the suite.

## Test each new framework first

Standalone suites let you test a framework without needing the others installed:

```bash
# A100 SGLang: inspect, then generate two prompts per workload.
bash scripts/results.sh infer --config suites/a100-qwen25-7b-sglang.json \
  --run-id a100-sglang-smoke --limit 2 --dry-run
bash scripts/results.sh infer --config suites/a100-qwen25-7b-sglang.json \
  --run-id a100-sglang-smoke --limit 2

# A100 TensorRT-LLM.
bash scripts/results.sh infer --config suites/a100-qwen25-7b-tensorrt.json \
  --run-id a100-tensorrt-smoke --limit 2

# MI210 SGLang, on the AMD host.
export DRIFTBENCH_PYTHON="$PWD/.venv-rocm-vllm/bin/python"
bash scripts/results.sh infer --config suites/mi210-qwen25-7b-sglang.json \
  --run-id mi210-sglang-smoke --limit 2
```

`--dry-run` validates the plan, not installed environments or kernels. The first two
prompts do not establish full context support. Before the full suite, run all long-context
prompts for each new standalone framework, using a new run ID, for example:

```bash
bash scripts/results.sh infer --config suites/a100-qwen25-7b-tensorrt.json \
  --run-id a100-tensorrt-context-check --workloads long_context
```

The client refuses silent truncation and checks the server's reported effective context.
TensorRT reserves at most 131,072 KV tokens in this profile; actual allocation depends on
available memory. Its memory fraction applies to free GPU memory, while vLLM/SGLang use
different memory-budget semantics. Equal numeric settings do not establish identical
execution. Chunking, scheduling, kernels, and internal numerical operations can still differ.

## Run the combined suites

Once each framework is ready, use a fresh full-run ID:

```bash
# A100: 3 frameworks, 6,852 responses total.
bash scripts/results.sh infer --config suites/a100-qwen25-7b-frameworks.json \
  --run-id a100-qwen25-7b-frameworks-r01

# MI210: 2 frameworks, 4,568 responses total.
bash scripts/results.sh infer --config suites/mi210-qwen25-7b-frameworks.json \
  --run-id mi210-qwen25-7b-frameworks-r01
```

These create one folder per framework under `settings/`. Servers run sequentially on
one accelerator; each stops before the next starts. Existing vLLM-only suites and launchers
are unchanged. If a combined run stops at a missing environment, prepare it and rerun the
same inference command with `--resume` on the original machine and output path.
Packing a combined suite requires inference to finish for all of its settings; an individual
complete setting can be packed separately if necessary.

For A100, after the combined inference completes:

```bash
bash scripts/results.sh stage safety results/runs/a100-qwen25-7b-frameworks-r01
bash scripts/results.sh pack results/runs/a100-qwen25-7b-frameworks-r01 \
  --output results/transfers/a100-qwen25-7b-frameworks-r01-safety.tar.gz
```

Transfer and unpack the archive plus checksum on your RTX system, then run `stage code`
and `stage final` on the imported suite root. The commands process every framework and
produce a combined `collection/rows.csv`. MI210 results first go through your fixed A100
safety host, following the same transfer steps.

## Compare frameworks after final scoring

Staged final scoring does not automatically run the suite's comparison declarations.
Compare the finalized setting directories explicitly, for example:

```bash
bash scripts/results.sh compare \
  results/runs/a100-qwen25-7b-frameworks-r01/settings/a100_qwen25_7b_vllm \
  results/runs/a100-qwen25-7b-frameworks-r01/settings/a100_qwen25_7b_sglang \
  --semantic --allow-confounded \
  --output results/comparisons/a100-qwen25-vllm-vs-sglang
```

Replace the second setting with `a100_qwen25_7b_tensorrt` for TensorRT, or use the MI210
setting paths for its vLLM/SGLang comparison. `--allow-confounded` records the differing
framework environments/engine settings; evaluator compatibility is still required.
