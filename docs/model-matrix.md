# Three models across A100, MI210 and Blackhole P150b

Each suite selects three models and two serving frameworks, vLLM and SGLang.
It runs servers sequentially and saves each setting separately. The existing
Qwen2.5 configurations are reused; earlier suites, including TensorRT, remain available.

| Model | Checkpoint | Prompt format |
|---|---|---|
| Qwen3.5 9B Base | `Qwen/Qwen3.5-9B-Base` | Raw published prompt |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | Official chat template |
| Llama3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Official chat template |

Every config pins an immutable Hub revision. Qwen3.5 uses the explicitly requested
[pretrained Base checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B-Base), rather than
the post-trained `Qwen/Qwen3.5-9B` checkpoint. Qwen2.5 and Llama use Instruct variants,
consistent with the earlier profiles. Llama weights require approved Hub access.

All settings use seed 42, greedy decoding, a 512-token output budget, a requested
32,768-token context, client batches of 16 and long-context batches of one. NVIDIA
and AMD request BF16; TT internal precision depends on its implementation and must
be recorded from the serving host. Each model's prompt settings are identical across
hardware and frameworks. Compare drift within the same model; Base-versus-Instruct
quality differences are a different experiment.

| Host | Suite | Serving environments | Validation status |
|---|---|---|---|
| A100 | `suites/a100-models-vllm-sglang.json` | `.venv`, `.venv-sglang` | Full-model hardware validation pending |
| MI210 | `suites/mi210-models-vllm-sglang.json` | `.venv-rocm-vllm`, `.venv-rocm-sglang` | ROCm/gfx90a kernel validation pending |
| Blackhole P150b | `suites/blackhole-p150b-models-vllm-sglang.json` | `.venv-tt-vllm`, `.venv-tt-sglang` | Experimental templates; support differs by model/framework |

The NVIDIA reference runtimes contain the Qwen3.5 architecture. This does not establish
MI210 support for its Gated DeltaNet kernels. Use ROCm builds for AMD and the TT
integrations for Tenstorrent; the NVIDIA installer lockfiles apply only to A100.
See the [framework environment guide](qwen25-frameworks.md).

Tenstorrent's [P150 model table](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/models_by_hardware.md)
lists Llama3.1 8B as experimental. Its
[model page](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/Llama-3.1-8B_p150.md)
includes Instruct weights and a TT-Metal vLLM deployment. The table does not list the
requested Qwen checkpoints on P150. The
[SGLang plugin README](https://github.com/tenstorrent/tt-inference-server/blob/main/tt-sglang-plugin/README.md)
describes testing on Wormhole N150/N300/T3K, without establishing Blackhole support.
Accordingly, all six P150b configs carry `experimental_unverified`; they reserve the
requested experiment settings but do not implement missing TT kernels. These support
sources were checked on 2026-09-14. No A100, MI210 or P150b model inference was run
when preparing this matrix.

## Inspect and smoke-test

Run each command on its corresponding host. Add `--dry-run` first to inspect the
resolved plan without loading a model. Each smoke run requests two prompts from each
of the five workloads, or 60 responses across six settings.

```bash
# A100
bash scripts/results.sh infer --config suites/a100-models-vllm-sglang.json \
  --run-id a100-models-smoke --limit 2

# MI210: use the ROCm client/runtime prepared on this host.
export DRIFTBENCH_PYTHON="$PWD/.venv-rocm-vllm/bin/python"
bash scripts/results.sh infer --config suites/mi210-models-vllm-sglang.json \
  --run-id mi210-models-smoke --limit 2

# P150b: inspect the experimental matrix before attempting supported settings.
export DRIFTBENCH_PYTHON="$PWD/.venv-tt-vllm/bin/python"
bash scripts/results.sh infer --config suites/blackhole-p150b-models-vllm-sglang.json \
  --run-id p150b-models-smoke --limit 2 --dry-run
```

On P150b, a whole-matrix attempt requires `--allow-experimental` instead of `--dry-run`
and will fail wherever the installed backend lacks support. Start with the upstream
documented Llama/vLLM combination by selecting it explicitly:

```bash
bash scripts/results.sh suite --config suites/blackhole-p150b-models-vllm-sglang.json \
  --output results/runs/p150b-llama-smoke \
  --settings blackhole_p150b_llama31_8b_instruct_vllm --limit 2 --inference-only
```

`suite --settings` also works for individual A100/MI210 settings. It leaves other
settings pending, so evaluate or pack the completed setting directory rather than
the whole suite. It does not load `.env` automatically; export `HF_TOKEN` in the
shell or use an existing authenticated Hugging Face session for gated weights.

## Full inference and portable evaluation

After model smoke tests and all long-context prompts pass, use a new run ID and omit
`--limit`. Each setting produces 2,284 responses; a complete six-setting host suite
produces 13,704. Smoke results should keep their own run ID.

For example, A100 saves:

```text
results/runs/a100-models-r01/
  suite.json
  settings/
    a100_qwen35_9b_base_vllm/
    a100_qwen35_9b_base_sglang/
    a100_qwen25_7b_vllm/
    a100_qwen25_7b_sglang/
    a100_llama31_8b_instruct_vllm/
    a100_llama31_8b_instruct_sglang/
  collection/
```

Use the existing [staged workflow](staged-workflow.md) to pack and transfer results.
All three new suites pin the same Llama-Guard-3-8B judge on `cuda`, intended for the
fixed A100 evaluation host after inference. MI210/TT results move there for safety
scoring; HumanEval execution stays on the RTX host with bubblewrap. `infer` performs
neither evaluation stage and requires no bubblewrap on the inference host.

The three comparison declarations in each suite pair vLLM and SGLang for the same
model. Staged final scoring does not execute these declarations automatically; run
`compare` explicitly after evaluation as shown in the
[framework comparison guide](qwen25-frameworks.md#compare-frameworks-after-final-scoring).
