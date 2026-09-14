# Accelerator and serving framework experiments

Use the same benchmark client, prompt files, tokenization and evaluation for all
targets. Each server runs in its own framework environment. CUDA, ROCm and
Tenstorrent Python packages must be installed separately.

The original 2,284-response NVIDIA/vLLM **offline** baseline remains in
`results/rtx3060-qwen35-vllm`. New HTTP experiments use separate directories.
An HTTP client batch is a group of concurrent requests followed by a barrier;
it does not promise the same internal GPU batch shape as the offline baseline.
Use HTTP for both sides of a new framework comparison.

| Target | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| NVIDIA RTX 3060 6 GB | Original full run; Qwen3.5/Qwen2.5 HTTP samples tested | Qwen3.5 sample and longest prompt tested | Qwen2.5 samples, longest prompt and 16-request client batch tested |
| AMD RX 7900 XT | ROCm profile, hardware pending | ROCm profile, hardware pending | Unsupported vendor |
| AMD MI210 | ROCm profile, hardware pending | ROCm profile, hardware pending | Unsupported vendor |
| Tenstorrent Wormhole | N150/Qwen3-8B vendor profile, hardware pending | N150/Llama3.1-8B plugin example, hardware pending | Unsupported vendor |
| Tenstorrent Blackhole | P300/Qwen3-8B vendor profile, hardware pending | Upstream support not established here | Unsupported vendor |

The user currently has no SSH access to the AMD/Tenstorrent machines. No results
from those devices are claimed. N150 and P300 are concrete example boards;
adapt the device mesh to the actual board before running.

## Client and common evaluator

The client does not need an accelerator or a serving-framework installation:

```bash
python3 -m venv .venv-client
.venv-client/bin/pip install -e '.[client,evaluation,test]'
.venv-client/bin/python -m driftbench_runner --help
```

Run from the project root. Config paths and output paths are relative to that
directory. `doctor` reports actual accelerators and package versions on the
host where it is run:

```bash
python -m driftbench_runner doctor --output results/hardware.json
```

## NVIDIA

The original vLLM environment is created by `scripts/bootstrap.sh`. The new
vLLM server profile is `configs/rtx3060-qwen35-vllm-http.json`.

Install SGLang and its workspace CUDA/GCC toolchains without `sudo`:

```bash
bash scripts/install_sglang.sh
```

The installer restores `requirements.sglang.lock.txt` into `.venv-sglang` and installs
the runner. See [installer requirements and checks](qwen25-frameworks.md#environments-and-support).

On a host with a suitable C++20 compiler and CUDA toolkit, remove the local
`workspace_gcc` and `cuda_version` launch settings and use that host's toolchain.
The selected compiler and runtime environment are recorded in server provenance.
The CUDA helper downloads only compiler, runtime and headers from NVIDIA's
redistribution manifest and checks the published SHA-256 before extraction.

Run a local server in terminal 1 and the benchmark client in terminal 2:

```bash
source scripts/env.sh
.venv-sglang/bin/python -m driftbench_runner serve \
  --config configs/rtx3060-qwen35-sglang-http.json

# Terminal 2, after "Server ready":
source scripts/env.sh
python -m driftbench_runner run \
  --config configs/rtx3060-qwen35-sglang-http.json \
  --output results/my-sglang-run --limit 2
```

Omit `--limit 2` to run all 2,284 published prompts. Use `--resume` only with
identical configuration, prompt set and serving environment. Stop the server
with Ctrl-C before running a GPU judge. Each server loads one model.

The convenience script starts the server, waits for a successful API generation,
runs all five workloads, stops its own server, and evaluates on the common host:

```bash
bash scripts/run_served.sh .venv/bin/python \
  configs/rtx3060-qwen35-vllm-http.json results/my-vllm-http --limit 2
bash scripts/run_served.sh .venv-sglang/bin/python \
  configs/rtx3060-qwen35-sglang-http.json results/my-sglang-http --limit 2
```

The script's safety judge uses CPU to keep its evaluation environment consistent
across accelerator experiments. The manual `judge-safety --device cuda` option
is available when evaluating both runs on the same NVIDIA evaluation machine.

The recorded environments are `requirements.sglang.lock.txt`,
`requirements.tensorrt.lock.txt`, and `requirements.gcc13.explicit.txt`.
The current TensorRT lock replaces the historical CUDA 13 stack with TensorRT-LLM 0.20.0/CUDA 12.8. Earlier validation reports describe their recorded environments; they do not validate the replacement stack.

### TensorRT-LLM

For the current **full-workload 6 GB preset**, use
`rtx3060-qwen25-05b-tensorrt-http.json` and its paired
`rtx3060-qwen25-05b-vllm-http.json`. They use Qwen2.5-0.5B-Instruct at the same
pinned revision. The TensorRT KV pool is bounded at 65,536 tokens (BF16), leaving
working memory available while requesting a 32K context. The launcher records
the observed limit and the client enforces it. The suite and current
row-comparison workflow are described in [the comparison guide](comparing-settings.md).

Here, “TensorRT serving” means NVIDIA **TensorRT-LLM**, using `trtllm-serve`.
The adapter supports its completions API. This is not Torch-TensorRT or Triton
Inference Server.

The previously validated TensorRT-LLM 1.2.1 registers `Qwen3ForCausalLM` but does not register
Qwen3.5. The explicit `rtx3060-qwen3-06b-tensorrt-http.json` profile uses
**Qwen3-0.6B** for compatibility testing. Comparing it to Qwen3.5 changes the
model and must not be described as framework-only drift.
The matching vLLM profile is `rtx3060-qwen3-06b-vllm-http.json`; it uses the same
Qwen3 revision, prompt template, greedy settings and client batch plan.

These Qwen3 profiles are retained as earlier experiments and require 1.2.1; the current 0.20.0 installer is for the Qwen2.5 profiles. A later launch on the
same laptop exposed only 20,448 context tokens, below the longest request's
22,416-token budget; increasing the cache then left insufficient working
memory. Use the Qwen2.5 pair for the default full run on this laptop.

```bash
bash scripts/install_tensorrt.sh
.venv-trt/bin/python -m driftbench_runner serve \
  --config configs/rtx3060-qwen25-05b-tensorrt-http.json
```

The launcher supplies revision, context and batch settings explicitly. Additional
TensorRT API options and dtype are written to a generated JSON/YAML file from the
experiment config and included in its fingerprint. The local launcher supplies
the isolated environment's CUDA library paths. Package installation alone does
not establish model or GPU support; consult the validation report.

TensorRT and TT receive a local alias to the immutable Hub snapshot. This pins
tokenizer and configuration files as well as weights. TensorRT exposes the local
directory basename as its model ID; `server.model_name` therefore explicitly
sets `Qwen3-0.6B` while the benchmark's model identity remains `Qwen/Qwen3-0.6B`.

TensorRT may round or reduce its KV-cache attention window. Release 0.20.0 has no
CLI fail-fast flag. The runner captures the worker's explicit `max_seq_len` log line
after KV-cache allocation and stores `effective_context_limit`, capped at the
requested context length.
The benchmark refuses prompts whose input plus generation budget exceeds that
observed limit. The longest Qwen3-0.6B request needs 22,416 tokens.

## AMD: RX 7900 XT and MI210

Use a working ROCm framework environment on the AMD host. The profiles are
`rx7900xt-qwen35-{vllm,sglang}-http.json` and
`mi210-qwen35-{vllm,sglang}-http.json`.
The installed NVIDIA wheels in this workspace cannot be used on AMD. Follow
the official [vLLM ROCm installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
and [AMD SGLang instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html)
for the board and supported OS. The families are gfx1100 (7900 XT) and gfx90a
(MI210). A working family-level ROCm stack does not by itself prove Qwen3.5 kernel support.

Copy this project to the host, install the runner with `pip install -e . --no-deps`
inside the serving environment, and run `doctor`, `serve --dry-run`, then `serve`
with the corresponding config. The launcher checks a usable HIP accelerator;
the client sends identical token IDs. NVIDIA management tools are optional.
The SGLang profiles select Triton attention and disable overlap scheduling.
The launcher refuses unmapped engine settings rather than ignoring them.

## Tenstorrent

Use the vendor's TT-Metal serving environment. Ordinary CUDA/ROCm vLLM wheels
cannot run on TT accelerators. The [official inference server](https://github.com/tenstorrent/tt-inference-server)
contains separate vLLM and SGLang integrations.

The two vLLM profiles derive board settings and container tags from the saved
official `release_model_spec.json` in `vendor/serving-references`:

| Profile | Model | Vendor container tag |
|---|---|---|
| `wormhole-n150-qwen3-8b-vllm-http.json` | Qwen3-8B | `0.10.0-e0e0500-409b1cd` |
| `blackhole-p300-qwen3-8b-vllm-http.json` | Qwen3-8B | `0.10.0-e867533-22be241` |

Full image names and release requirements are stored in each profile's
`validation` field. Use the vendor's host setup/device mounts for the actual
board. Blackhole P300 requires the release's matching KMD and firmware. The
profiles select TT-specific block size, additional configuration, architecture
and mesh settings; they are still awaiting physical validation.

The release registry does not list Qwen3.5-0.8B. Qwen3-8B is an **explicit separate
experiment**, not a silent substitution. Compare it to Qwen3-8B at the same
Hub revision on another sufficiently large GPU. TT internal weight/activation
precision may differ from the HF checkpoint dtype; inspect and record it before
interpreting numerical drift.

The SGLang example `wormhole-n150-llama31-8b-sglang-http.json` uses the vendor's
`sglang-tt-server` launcher. Install it using the
[official plugin instructions](https://github.com/tenstorrent/tt-inference-server/tree/main/tt-sglang-plugin)
inside its own **CPU PyTorch + TTNN** environment. The README documents Llama3.1-8B
and Wormhole boards; it does not establish Blackhole support. Llama needs approved
HF access, which the user does not currently have. The generic SGLang protocol
adapter is shared; device support comes from the TT plugin.
The inspected plugin registry maps `Qwen2ForCausalLM`, not Qwen3 or Qwen3.5;
do not change this example to Qwen3 without a corresponding upstream implementation.

For an ungated option on **Wormhole N300**, use
`wormhole-n300-qwen25-7b-sglang-http.json` or its matching vLLM profile. These use
Qwen2.5-7B-Instruct, which matches the plugin's Qwen2 architecture registration.
All 2,284 prompts pass client preflight (maximum required context 22,433 tokens).
This is a board-specific template; it still needs validation on the actual N300.

For TT, the launcher downloads the exact Hub snapshot and passes its local path
to the backend. This also avoids relying on a plugin to apply a floating model
revision correctly. All profiles retain a human-readable served model alias.

## Remote servers and trustworthy comparisons

Launch on the accelerator host first. Copy its ready server metadata JSON to the
benchmark client, keeping the filename configured in `server.metadata_path`.
Change only `server.base_url` to the reachable HTTP root (no `/v1` suffix).
Use the same model revision, engine, hardware and launch configuration in both
copies. An SSH tunnel can keep the server bound to loopback when access becomes
available. For authenticated existing endpoints, set `server.api_key_env` to a
locally defined environment variable name; do not put tokens in JSON.

`serve` captures observed packages, CUDA/HIP runtime, GPU names, PCI IDs, TT
device nodes, launch argv and an instance ID on the **serving host**. `run` binds
these observations to the model/config fingerprint and separately records the
client environment. The sidecar is trusted operator provenance, not cryptographic
remote attestation. `/v1/models` checks the exposed alias, not weight contents.
Do not replace or hot-swap the server during a benchmark. Inference failures
remain failed runs; there is no silent request retry.

The output contains exact sent input IDs and, when the framework returns them,
actual generated IDs. Some older TT vLLM APIs do not return generated IDs. Those
records use `null`, and token drift is unavailable; response text is never
retokenized and passed off as generated IDs. Full server responses and per-request
latency are retained. HTTP timing includes transport and scheduling overhead and
does not measure TTFT. Partially saved batches are resent together on resume.

Evaluate both inference directories on a common host, with the same small
safety judge or the reserved LlamaGuard-3-8B option, then compare:

```bash
python -m driftbench_runner compare results/my-vllm-http results/my-sglang-http \
  --semantic --allow-confounded --output results/framework-comparison
```

Framework packages, kernels and engine knobs can differ, so the comparison
records those differences and requires explicit acknowledgment. Hardware-only
comparisons should keep model, framework versions, decoding, precision, client
batch sizes, engine settings and transport fixed. A two-example validation run
checks integration; it is not a reliable estimate of population drift.
