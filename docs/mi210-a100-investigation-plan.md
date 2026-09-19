**Investigation plan: why do MI210 and A100 produce different accuracy?**

Objective: explain reproducible answer differences, distinguishing evaluation effects, within-device variability, and differences between serving implementations. A100 is the comparison baseline, not numerical ground truth. This is a plan; new accelerator experiments have not been run.

Implementation and commands: [Running the investigation](mi210-a100-investigation.md), using `bash scripts/investigate.sh`.

**Evidence already available — checked 2026-09-19**

The [saved comparison report](../results/comparisons/a100-baseline-20260916/README.md) covers five matching model checkpoints. MI210 is not uniformly worse:

| Model | A100 math accuracy | MI210 math accuracy | Math flips | A100 correct → MI210 incorrect | Reverse |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B Instruct | 45.80% | 48.60% | 9.20% | 16 | 30 |
| Qwen3.5 0.8B | 48.80% | 47.80% | 10.60% | 29 | 24 |
| Llama 3.1 8B Instruct | 80.00% | 81.60% | 4.80% | 8 | 16 |
| Qwen2.5 7B Instruct | 89.60% | 89.60% | 1.60% | 4 | 4 |
| Qwen3.5 9B Base | 68.40% | 68.60% | 6.60% | 16 | 17 |

Each math comparison contains 500 paired prompts. For code and math together, all 3,320 saved A100–MI210 pairs have identical rendered prompts, input token IDs, and effective sampling settings. Both sides have saved output token IDs. This check establishes agreement in the recorded requests, not equivalence of all internal server processing.

Saved manifests show BF16, temperature 0, seed 42, tensor parallelism 1, and matching generation settings. The recorded environments differ: A100 uses vLLM 0.17.1 / PyTorch 2.10.0; MI210 uses vLLM 0.17.1+rocm700 / PyTorch 2.9.1+git8907517. Small-model GPU memory utilization is 0.55 versus 0.80; larger-model engine configuration dictionaries match. The same utilization fraction on different VRAM capacities would not establish identical scheduling. Inspect actual launches, resolved kernel choices, and available KV-cache capacity.

**1. Classify the existing disagreements before rerunning models**

- Use all five models' existing code/math pairs to count regressions, improvements, changed outputs with unchanged correctness, and unchanged controls.
- Review saved extraction, finish reasons, output lengths, and references. Assign categories: substantive answer change, code/answer extraction, output-limit effect, ambiguous reference, or unresolved.
- Starting examples: Qwen0.8B `gsm8k_180` (5 versus 15), Llama1B `gsm8k_418` (30 versus 26), Qwen0.8B `humaneval_162` (first code-block extraction), and Qwen9B `gsm8k_065` (correct initial answer followed by an unrelated extracted number). Read full outputs and references before classifying.
- Keep original benchmark scores. Any alternative extraction/scoring is a separate sensitivity analysis, not a replacement result.
- First-week diagnostic set: 24 cases per small model, Qwen0.8B and Llama1B: eight regressions, eight improvements, eight unchanged-output controls, covering code and math. Select a mix of lengths and finish reasons; save the selection rule and IDs before running experiments.
- Deliverable: a frozen case manifest and classification table. This deliberately selected subset cannot estimate population accuracy; validate any general improvement on the full code/math workloads later.

**2. Establish the within-device variability baseline**

- Repeat the 48 selected cases three times on each device using the original configurations, fixed request order, seed, and generation limits. Include at least one fresh server launch on each device. Initial budget: 288 generated responses.
- Preserve model and tokenizer revisions, prompt rendering, input IDs, effective request settings, output IDs, finish reasons, server metadata, batch grouping, and environment information for every trial.
- Compare same-device repeat mismatches with cross-device mismatches. If the same device changes answers, investigate scheduling/execution variability before assigning the difference to a particular device.
- Score all code/math outputs on the same evaluator environment, using the existing isolated code-evaluation workflow. GPU hosts need not execute generated code.
- Deliverable: within-device and cross-device token/label agreement, plus the subset of repeatable cross-device failures.

**3. Test serving factors one at a time**

| Experiment | Change from its control | What it tests |
|---|---|---|
| Serial execution | One client request in flight and engine maximum running sequences 1 | Sensitivity to batching/scheduling |
| Eager execution | With serial execution held fixed, change only graph/compilation execution controls supported by that backend | Sensitivity to optimized execution paths |
| Precision diagnostic | On a few stable cases or isolated operations, increase accumulation/intermediate precision where supported | Sensitivity to numerical precision |

- Preserve model weights, tokenization, sampling, stop rules, and output caps across each matched experiment. Record all flags an execution control actually changes; eager execution is not a guarantee of determinism or a shared kernel implementation.
- Inspect actual attention/matmul implementations rather than inferring them from matching framework names. Attempt a common compatible software release only as a separate environment experiment, with working original environments preserved. Do not force incompatible CUDA and ROCm packages into nominal version parity.
- Match explicit sequence/token budgets and avoid memory pressure. Test the small-model memory allocation difference separately if resolved cache/scheduling information suggests it matters.
- Repeat the most informative conditions; avoid a full combinatorial sweep during week one.
- Deliverable: an experiment matrix showing which intervention changes reproducibility, agreement, and correctness. A changed result narrows the hypothesis; it does not by itself identify a faulty instruction.

**4. Locate the first consequential numerical difference**

- Use saved output token IDs to find each pair's longest common prefix and first different token, including termination/EOS differences.
- Replay the exact input tokens plus that shared prefix on both devices, and examine the next-token predictions. Do not compare later free-running logits after the histories have diverged.
- Collect top-k token IDs and scores, top-1/top-2 margins, and full logits if the actual serving backend exposes them. API log-probabilities are not raw logits; truncated top-k responses do not provide full-vocabulary logit error. Record the score type and unavailable measurements explicitly.
- For a small number of stable cases, capture corresponding layer/operator inputs and outputs. Compare tensor bits only after matching dtype, shape, and logical element ordering; also report numerical error. A separate reference forward pass is a diagnostic, not proof about the original vLLM kernel.
- If a particular operation is implicated, reproduce it with captured inputs in a microbenchmark. Test accumulation order, conversion boundaries, and intermediate precision against an independently checked higher-precision reference, then see whether the intervention changes the original model case.
- Deliverable: one or more evidence chains from operation/tensor difference → ranking/token change → scored outcome, or a precise account of where instrumentation prevents further attribution.

**Five agreement measurements and their availability**

| Measurement | Planned evidence |
|---|---|
| Bitwise | Exact bits of corresponding captured tensors; identical bits are not an assumed cross-device requirement |
| Logit | Same-prefix next-token vectors, with the error/normalization definition recorded; requires instrumentation |
| Top-k | Candidate overlap, rank changes, and score margins under the same prefix; check API support first |
| Token | Exact sequence agreement and first differing position; saved server token IDs already exist |
| Validation accuracy | Correctness and paired flips, in both directions, under one evaluator |

Floating-point implementations can differ with reduction order, batching, platform, and library version. That makes these plausible mechanisms to investigate, not established causes of these runs. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

The pinned vLLM 0.17.1 documentation does not promise default reproducibility or cross-hardware equality. Its reproducibility controls have mode-specific behavior; verify backend support before trying batch-invariance settings, especially on ROCm. [vLLM 0.17.1 reproducibility](https://docs.vllm.ai/en/v0.17.1/usage/reproducibility/).

**One-week scope and completion criteria**

| Work | Estimated effort | Output |
|---|---|---|
| Saved-result triage and manifest review | 1 day | Classified cases and frozen 48-case subset |
| Repeated baseline and serial experiments | 1–2 days | Within-device versus cross-device variability |
| Targeted execution/precision experiments and first-token analysis | 1–2 days | Leading explanations with supporting cases |
| Synthesis | 0.5–1 day | Findings, limitations, and reproducible follow-up cases |

The schedule assumes access to both allocated GPU nodes and the working serving environments. Raw-logit/layer instrumentation or reproducing an unsupported kernel may extend beyond one week; token-level analysis and scoring triage do not depend on that instrumentation. Reproducing and classifying the effect is the minimum useful deliverable; one isolated numerical mechanism is a stretch goal, not a promised hardware fix.

Keep chat out of scope. Defer safety attribution until saved answers can be labeled with one fixed judge runtime; its current runtime differences confound the labels. Investigate long-context scoring separately before adding it to numerical diagnosis. PD/communication performance is a separate study.

Store new experiments under a new `results/investigations/mi210-a100/` directory, grouped by model, device, condition, and repeat. Keep the original benchmark runs immutable. End with a table of observation, hypothesis, intervention, outcome, and remaining uncertainty; do not claim a hardware defect from output inequality alone.
