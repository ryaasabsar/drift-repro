# Serving framework validation — 9 September 2026

Real inference was run on the local NVIDIA RTX 3060 Laptop GPU, 6 GB, under
WSL2. Each HTTP sample run contains the first **two prompts from each of the five
published workloads**: 10 responses. These are integration checks, not a full
replication or reliable estimates of model accuracy or hardware drift.

| Model | Framework | Evidence |
|---|---|---|
| Qwen3.5-0.8B | vLLM 0.17.1 | [10 HTTP responses and evaluations](../results/validation-vllm-http/manifest.json); original full 2,284-response baseline also retained |
| Qwen3.5-0.8B | SGLang 0.5.10.post1 | [10 HTTP responses and evaluations](../results/validation-sglang-http/manifest.json), plus [longest prompt](../results/validation-sglang-longest/manifest.json): 21,688 input tokens |
| Qwen3-0.6B | TensorRT-LLM 1.2.1, PyTorch backend | [10 HTTP responses and evaluations](../results/validation-tensorrt-http/manifest.json), plus [longest prompt](../results/validation-tensorrt-longest/manifest.json): 21,904 input tokens |
| Qwen3-0.6B | vLLM 0.17.1 | [10 HTTP responses and evaluations](../results/validation-vllm-qwen3-http/manifest.json); complete `run_served.sh` workflow tested |

The safety evaluator for these samples is pinned Qwen3Guard-Gen-0.6B on the
common CPU host, in FP32. HumanEval uses the same isolated evaluator as the
original baseline. GSM8K exact match and Qasper F1 are unchanged. Chat is evaluated
through paired semantic comparison. The LlamaGuard-3-8B option remains available
for later use with approved model access and sufficient memory.

The [Qwen3.5 vLLM/SGLang comparison](../results/validation-vllm-vs-sglang/comparison.json)
contains per-prompt text/token changes, correctness and safety label flips, chat
embedding drift, confidence intervals and all changed experimental controls.
In this small sample, **7/10 texts changed and 0/8 objective labels flipped**.
Both chat pairs required embedding-input truncation; this is explicitly recorded.
The framework/library versions, attention kernels and engine controls differ,
and the report is labeled with those confounds.

The [Qwen3 vLLM/TensorRT comparison](../results/validation-vllm-vs-tensorrt/comparison.json)
uses the same Qwen3-0.6B model revision on both sides. It also has **7/10 texts
changed and 0/8 objective labels flipped**. Token comparisons are unavailable
because TensorRT omitted generated IDs. This sample has two chat pairs with no
embedding truncation. Both comparisons remain small, confounded integration
examples; no GPU-only drift claim is made.

vLLM and SGLang returned actual generated token IDs. This TensorRT API returned
text and token counts but omitted generated IDs. Its records store `null` and
token drift remains unavailable; the runner does not manufacture IDs by
retokenizing response text.

TensorRT 1.2.1 does not register the Qwen3.5 architecture, so its explicit model
is Qwen3-0.6B. Its launcher pins the entire local Hub snapshot to keep the
tokenizer/configuration from following `main`. It captures the worker's actual
context capacity and checks every run's input-plus-output budget against that
capacity. Both the 10-response run and longest-prompt run used this final path.
Earlier failed startup attempts and the initial compatibility run are separate
from the final evidence.

SGLang required workspace CUDA 12.8 and GCC 13 to build its C++20 kernels.
TensorRT required CUDA 13 PyTorch, MPI, the CUDA 13 compiler components and
explicit library/header paths. These are isolated from the original vLLM
environment. Exact package/compiler locks and launch settings are included.

**44 tests passed**, including protocol controls, missing-token handling,
effective context limits, resume/provenance checks and isolated HumanEval
execution. All three inference ports were closed after validation. The final
validation set contains 42 generated responses across four sample runs and two
longest-prompt checks, in addition to the original full baseline.

AMD 7900 XT / MI210 and Tenstorrent Wormhole / Blackhole have profiles and
framework-specific launch support, but **no physical validation**: the user
currently has no SSH access or endpoints. Client preflight succeeds for all
2,284 prompts with the ungated TT Qwen3-8B and Qwen2.5-7B-Instruct profiles.
Preflight and protocol tests do not establish accelerator compatibility.
See [the platform guide](platforms.md) for supported combinations and assumptions.

The original baseline archive is unchanged, SHA-256:
`8faf90dcbe0618f7f7af83d32b172b78463a28dff82908e8e46523a781472fb4`.
