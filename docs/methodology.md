# Experimental contract

Production inference covers A100 40 GB, MI210 and Blackhole P150b with the three pinned models in `configs/`. vLLM serves all targets; SGLang serves NVIDIA and AMD. The goal is to preserve outputs and measure behavioral changes under another setup. A single run measures task performance; it cannot establish cross-hardware drift by itself.

## Sources and pins

- [Paper](https://openreview.net/forum?id=Xfzzp6grRP): *DriftBench: Measuring and Predicting Infrastructure Drift in LLM Serving Systems*.
- [Artifact](https://github.com/GianluigiVitale/driftbench-ae), commit `c915e781a17c2d4c9bd768e60a1bc9734c5f2895`.
- Model revisions: `models.lock.json` and the matching per-setting configs (Qwen3.5-9B-Base, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct).
- [vLLM Qwen recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html); installed vLLM 0.17.1, using its Qwen3.5 implementation and text-only option.
- [Paper safety judge](https://huggingface.co/meta-llama/Llama-Guard-3-8B); the fixed safety evaluator on A100.

## What matches and what changes

We preserve every original prompt and ground truth in the published files. All HumanEval tasks, the published 500 GSM8K tasks, 520 AdvBench tasks and 100 Qasper tasks are used. Chat uses all 1,000 published prompts because the paper's 973-item subset is not specified by this artifact.

The original reproduction script defaults to greedy sampling, top-p 1, 512 new tokens and seed 42. We retain these controls. Its vLLM path passes raw strings to `generate`. Qwen3.5 Base receives raw published prompts; Qwen2.5 and Llama Instruct receive their official chat templates. The exact rendered strings and token IDs are recorded. Do not mix prompt formats when attributing a difference to hardware.

This is an adaptation of the paper's experiment. NVIDIA/AMD profiles request native BF16. TT internal weight/activation precision comes from its model implementation and must be recorded from the serving host. No quantized replacement weights or CPU fallback are silently substituted.

Profiles request 32,768 tokens of context. Input lengths depend on the selected model tokenizer and template; the runner checks the full selected input plus output budget before inference. The preflight check is repeated before inference and refuses overflow instead of trimming context or shortening response budgets per prompt.

Production profiles submit one request at a time for every workload and limit the server to one running sequence. This reduces scheduling variation at the cost of throughput. Framework-specific prefill/cache controls are recorded explicitly. The local RTX smoke profiles are separate archived checks, not production comparison settings. Batch membership is reconstructed from the full input list during resume; a partially written batch is reissued in full while preserving previously saved outputs. Always compare runs using the same batching profile.

Safety evaluation uses the pinned Llama-Guard-3-8B on one A100 environment after inference finishes. It uses greedy decoding, one beam, eager attention, seed 42 and strict PyTorch deterministic algorithms with TF32 disabled. Unsupported deterministic operations fail explicitly. These controls are included in judge provenance; old labels must not be mixed with labels produced under this policy. It receives both the source user prompt and saved model response through its official template. Archived Qwen3Guard labels use a different evaluator and cannot be mixed with these labels for drift comparisons.

Safety-judge provenance also includes its CPU/GPU hardware, Python, PyTorch, Transformers and CUDA runtime. Re-evaluate both target runs on a common judge host when changing inference hardware. Otherwise, changes in the judge's own kernels could be mistaken for changes in the inference model. The comparison refuses mismatched judge provenance even with `--allow-confounded`.

## Scoring

HumanEval uses one completion per problem. Code fences/full functions and function-body continuations are supported; original tests execute in an isolated Python process. Evaluator v3 preserves helper functions supplied in the source prompt, accepts complete Python inside an unclosed Markdown fence, and compiles full-function answers separately in the same namespace. It does not repair or finish truncated Python. Generated `__main__` examples are not run as the benchmark tests, consistent with the [official HumanEval execution namespace](https://github.com/openai/human-eval/blob/master/human_eval/execution.py). Each worker initializes Python's random generator and hash seed to 42. Syntax errors, failed assertions, execution errors and timeouts are failures; unavailable sandbox infrastructure is pending, not an incorrect model answer.

The code evaluator records its Python version, operating system, CPU and resource limits. Evaluate both hardware runs on a common test host: changes in the Python runtime or timeout behavior must not be attributed to model inference. Comparison refuses mismatched code-execution environments and identical responses with different correctness labels.

GSM8K uses the final boxed/`####` answer if present, otherwise the final numeric expression. Commas and decimal formatting normalize with decimal arithmetic; no number is incorrect. This extraction policy is saved under a versioned evaluator. The 512-token cap may end an answer early; length-limited responses remain in denominators and are reported separately.

Qasper uses the maximum normalized English token-overlap F1 across accepted answers. Lowercasing, punctuation/articles and whitespace normalize, including special handling for yes/no/unanswerable. F1 ≥ 0.5 is the binary correctness rule used for flips.

Chat has no fabricated single-run accuracy. Pairwise scoring uses CPU `sentence-transformers/all-mpnet-base-v2` normalized embeddings. Semantic shift is `1 − cosine_similarity`; a shift > 0.3 is substantial, following the paper. The embedding evaluator uses seed 42, evaluation mode, one text per batch and strict PyTorch deterministic algorithms. Embedding model/revision, sequence limit, reproducibility controls and any encoder truncation are recorded. Length ratios and exact text changes accompany semantic scores. A short embedding context is distinct from model inference truncation.

Objective flip rate counts paired binary outcome changes, including both improvement and degradation. It is not the difference between aggregate accuracies and not the fraction of different strings. Wilson intervals use the number of valid scored pairs; unscored pairs stay visible. Raw strings and IDs are never dropped to make two runs align.

## Reproducibility limits

Every profile specifies seed 42, temperature 0, top-p 1, top-k -1 (disabled), min-p 0, repetition penalty 1, presence/frequency penalties 0 and a 512-token output cap. Launch scripts set Python hash seed 42; NVIDIA launches set the cuBLAS deterministic workspace configuration. vLLM uses its framework generation defaults and explicit request controls; SGLang uses OpenAI sampling defaults plus explicit request controls, avoiding hidden model sampling recommendations. SGLang also enables its native [deterministic inference mode](https://docs.sglang.io/docs/advanced_features/deterministic_inference), with the configured Triton attention backend. Support still needs validation on each target runtime/model.

vLLM 0.17.1 [batch invariance](https://docs.vllm.ai/en/v0.17.1/features/batch_invariance/) requires NVIDIA compute capability 9.0 or later, so that mode is not enabled on A100, RTX, MI210 or TT profiles. One-at-a-time serving reduces variability but is not an equivalent guarantee.

Temperature zero and a fixed seed do not guarantee bitwise-identical GPU results. Batch scheduling, CUDA, PyTorch, vLLM kernels and hardware can change outputs. Record these, keep the experimental controls fixed, and use repeated runs when estimating within-setup variability. HTTP request timing includes transport and scheduling overhead; it does not measure TTFT. The production client issues requests serially; this does not make the frameworks' numerical kernels equivalent. Start new run IDs after changing controls, and do not compare old batch-16 runs against these runs as a hardware-only change.

No PRI prediction or paper headline flip rate is extrapolated to these new runs. A later hardware comparison requires actual results from both machines.

Local validation of these controls used the cached Qwen3.5-0.8B checkpoint on RTX 3060, one GSM8K prompt, a 32-token cap and two independent server starts per framework. SGLang 0.5.10.post1 repeated the same token sequence with deterministic inference enabled; vLLM 0.17.1 produced different sequences despite identical input tokens and sampling controls. The CPU embedding evaluator repeated exactly on two short texts. These are small diagnostic checks, not evidence of determinism for the production 7–9B models or other accelerators. Their artifacts are in `.archive/check-output/rtx-smoke-20260914T173240Z/` and `.archive/check-output/determinism-embedding.json`.
