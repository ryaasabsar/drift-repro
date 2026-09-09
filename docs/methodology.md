# Experimental contract

The baseline is fresh inference on the RTX 3060 Laptop GPU, Qwen3.5-0.8B and vLLM. The goal is to preserve outputs so a later run can measure behavioral changes under another setup. A single run measures task performance; it cannot establish cross-hardware drift by itself.

## Sources and pins

- [Paper](https://openreview.net/forum?id=Xfzzp6grRP): *DriftBench: Measuring and Predicting Infrastructure Drift in LLM Serving Systems*.
- [Artifact](https://github.com/GianluigiVitale/driftbench-ae), commit `c915e781a17c2d4c9bd768e60a1bc9734c5f2895`.
- [Official Qwen model](https://huggingface.co/Qwen/Qwen3.5-0.8B), revision `2fc06364715b967f1860aea9cf38778875588b17`.
- [vLLM Qwen recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html); installed vLLM 0.17.1, using its Qwen3.5 implementation and text-only option.
- [Smaller safety judge](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B); its resolved revision is saved with each label.
- [Paper safety judge](https://huggingface.co/meta-llama/Llama-Guard-3-8B); gated, optional here.

## What matches and what changes

We preserve every original prompt and ground truth in the published files. All HumanEval tasks, the published 500 GSM8K tasks, 520 AdvBench tasks and 100 Qasper tasks are used. Chat uses all 1,000 published prompts because the paper's 973-item subset is not specified by this artifact.

The original reproduction script defaults to greedy sampling, top-p 1, 512 new tokens and seed 42. We retain these controls. Its vLLM path passes raw strings to `generate`. This adaptation applies Qwen's official user chat template without changing the source text, disables thinking, and records the exact rendered string and token IDs. Set `prompt_format` to `raw` for a separate raw-completion experiment. Do not mix prompt formats when attributing a difference to hardware.

This new model is outside the original paper's model matrix. Native BF16 is explicit; it is not the paper's FP16 baseline or an FP8/FP4 quantization experiment. No quantized replacement weights or CPU fallback are silently substituted.

The maximum observed rendered input was 21,688 tokens; the fixed 512-token response budget requires 22,200 total. The 32,768-token context handles every published input. The preflight check is repeated before inference and refuses overflow instead of trimming context or shortening response budgets per prompt.

The smoke profile runs one sequence at a time. The full-run profile uses fixed batches of 16 within each short workload and single requests for long context, with up to 2,048 prefill tokens per iteration. Batch membership is reconstructed from the full input list during resume; a partially written batch is reissued in full while preserving previously saved outputs. Always compare runs using the same batching profile.

The smaller safety judge is intentionally different from the paper's LlamaGuard-3-8B. It receives both user prompt and model response through its official moderation template. It classifies the response, not just whether the user asked for something harmful. `Controversial` remains visible in raw labels and is mapped to unsafe only for the explicitly named binary policy. Never compare one run scored by this judge against another scored by LlamaGuard without re-evaluating both using a common judge.

Safety-judge provenance also includes its CPU/GPU hardware, Python, PyTorch, Transformers and CUDA runtime. Re-evaluate both target runs on a common judge host when changing inference hardware. Otherwise, changes in the judge's own kernels could be mistaken for changes in the inference model. The comparison refuses mismatched judge provenance even with `--allow-confounded`.

## Scoring

HumanEval uses one completion per problem. Code fences/full functions and function-body continuations are supported; original tests execute in an isolated Python process. Evaluator v3 preserves helper functions supplied in the source prompt, accepts complete Python inside an unclosed Markdown fence, and compiles full-function answers separately in the same namespace. It does not repair or finish truncated Python. Generated `__main__` examples are not run as the benchmark tests, consistent with the [official HumanEval execution namespace](https://github.com/openai/human-eval/blob/master/human_eval/execution.py). Each worker initializes Python's random generator and hash seed to 42. Syntax errors, failed assertions, execution errors and timeouts are failures; unavailable sandbox infrastructure is pending, not an incorrect model answer.

The code evaluator records its Python version, operating system, CPU and resource limits. Evaluate both hardware runs on a common test host: changes in the Python runtime or timeout behavior must not be attributed to model inference. Comparison refuses mismatched code-execution environments and identical responses with different correctness labels.

GSM8K uses the final boxed/`####` answer if present, otherwise the final numeric expression. Commas and decimal formatting normalize with decimal arithmetic; no number is incorrect. This extraction policy is saved under a versioned evaluator. The 512-token cap may end an answer early; length-limited responses remain in denominators and are reported separately.

Qasper uses the maximum normalized English token-overlap F1 across accepted answers. Lowercasing, punctuation/articles and whitespace normalize, including special handling for yes/no/unanswerable. F1 ≥ 0.5 is the binary correctness rule used for flips.

Chat has no fabricated single-run accuracy. Pairwise scoring uses CPU `sentence-transformers/all-mpnet-base-v2` normalized embeddings. Semantic shift is `1 − cosine_similarity`; a shift > 0.3 is substantial, following the paper. Embedding model/revision, sequence limit and any encoder truncation are recorded. Length ratios and exact text changes accompany semantic scores. A short embedding context is distinct from model inference truncation.

Objective flip rate counts paired binary outcome changes, including both improvement and degradation. It is not the difference between aggregate accuracies and not the fraction of different strings. Wilson intervals use the number of valid scored pairs; unscored pairs stay visible. Raw strings and IDs are never dropped to make two runs align.

## Reproducibility limits

Temperature zero and a fixed seed do not guarantee bitwise-identical GPU results. Batch scheduling, CUDA, PyTorch, vLLM kernels and hardware can change outputs. Record these, keep the experimental controls fixed, and use repeated runs when estimating within-setup variability. The laptop also shares its GPU with display workloads, so timing measurements are descriptive rather than controlled performance claims.

No PRI prediction or paper headline flip rate is extrapolated to Qwen3.5-0.8B/RTX3060. A later hardware comparison requires actual results from both machines.
