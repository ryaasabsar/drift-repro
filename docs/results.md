# RTX 3060 baseline results

The complete run saved **2,284 responses** on an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM, using `Qwen/Qwen3.5-0.8B` and vLLM 0.17.1. Inference finished on 2026-09-09. The final integrity audit passed with **1,284 objective scores, 520 safety judgments and no missing records**. Chat has 1,000 saved responses for a later paired comparison.

| Workload | Responses | Result | Responses reaching 512-token cap |
|---|---:|---|---:|
| HumanEval | 164 | Pass@1: **37/164 (22.56%)** | 100 |
| GSM8K | 500 | Exact-match accuracy: **252/500 (50.40%)** | 66 |
| AdvBench | 520 | Classified safe: **466/520 (89.62%)** | 162 |
| LMSYS chat | 1,000 | Saved; semantic drift requires another run | 410 |
| LongBench Qasper | 100 | Mean token F1: **0.1506**; F1 ≥ 0.5 on **6/100** | 2 |

Safety was judged with **Qwen/Qwen3Guard-Gen-0.6B**, running in BF16 on the GPU. Its original labels were 466 safe, 46 unsafe and 8 controversial. The declared binary policy groups unsafe and controversial, producing 54/520 unsafe labels. This is a judge classification rate, not an accuracy measurement against human annotations. LlamaGuard-3-8B remains an optional judge; these labels are not presented as LlamaGuard results.

The profile uses native BF16, greedy decoding, seed 42, Qwen's user chat template with thinking disabled, and a 512-token response cap. Short workloads use fixed batches of 16; Qasper runs one request at a time. Every input fits the configured 32,768-token context. Length-limited responses remain in the reported denominators.

The published artifact contains 1,000 chat prompts, while the paper reports 973 without a recoverable subset list. This run uses all 1,000. The model, precision, chat formatting and smaller safety judge are explicit adaptations; see [methodology](methodology.md). These results do not reproduce the paper's large-model scores or establish cross-hardware drift.

Another inference job shared the GPU during much of the run. Recorded batch durations are useful execution records but should not be used as exclusive-GPU throughput measurements.

## Files

- [HumanEval responses](../results/rtx3060-qwen35-vllm/code.jsonl)
- [GSM8K responses](../results/rtx3060-qwen35-vllm/math.jsonl)
- [AdvBench responses](../results/rtx3060-qwen35-vllm/safety.jsonl) and [safety judgments](../results/rtx3060-qwen35-vllm/safety-labels.jsonl)
- [Chat responses](../results/rtx3060-qwen35-vllm/chat.jsonl)
- [Qasper responses](../results/rtx3060-qwen35-vllm/long_context.jsonl)
- [Per-prompt evaluation](../results/rtx3060-qwen35-vllm/evaluation.json), [summary CSV](../results/rtx3060-qwen35-vllm/summary.csv), [run manifest](../results/rtx3060-qwen35-vllm/manifest.json) and [integrity audit](../results/rtx3060-qwen35-vllm/audit.json)

The [portable bundle](../results/driftbench-rtx3060-bundle.tar.gz) includes the runner, configurations, dependency/model pins, upstream artifact and these results. Python environments, model weights and mutable caches are excluded; `scripts/bootstrap.sh` prepares an environment after extraction. The bundle's checksum is in [SHA256SUMS](../results/driftbench-rtx3060-bundle.sha256).

For a later A100 run, follow [the comparison instructions](../README.md#compare-with-an-a100-or-another-setup). Keep model revision, precision, prompts, decoding and batching fixed. Bring both output directories to the same evaluation host before running the paired comparison.
