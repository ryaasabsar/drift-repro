# Settings workflow validation

The current default suite completed on the RTX 3060 Laptop 6 GB with **two
prompts from each of the five workloads for each setting**: 40 responses,
32 objective evaluations, and 8 chat responses used in paired comparisons.
An additional TensorRT check processed the longest published prompt and a
16-request client batch. All 17 additional responses were evaluated.

| Setting | Framework version | Saved sample responses |
|---|---|---:|
| Qwen3.5-0.8B / vLLM | 0.17.1 | 10 |
| Qwen3.5-0.8B / SGLang | 0.5.10.post1 | 10 |
| Qwen2.5-0.5B-Instruct / vLLM | 0.17.1 | 10 |
| Qwen2.5-0.5B-Instruct / TensorRT-LLM | 1.2.1 | 10 |

These samples validate the workflow; they do not establish population drift
rates. The original full Qwen3.5/vLLM baseline is separately exported with all
2,284 responses. The default full four-setting experiment would contain 9,136
requests and was not run as part of this workflow check.

## Open the results

- [Four-setting matrix](../results/current-settings/matrix.csv): one prompt per row, responses and outcomes for every setting.
- [Matrices by workload](../results/current-settings/workloads): code, math, safety, chat and long context.
- [Qwen3.5 side-by-side viewer](../results/current-comparisons/qwen35-vllm-sglang/rows.html).
- [Qwen2.5 side-by-side viewer](../results/current-comparisons/qwen25-vllm-tensorrt/rows.html).
- [Full baseline rows](../results/rtx3060-qwen35-vllm/tables/rows.csv): 2,284 responses.
- [Combined collection](../results/collected-settings/matrix.csv): full baseline plus four sample settings; absent sample responses are `missing`.
- [Machine-readable audit](../results/settings-current-validation.json), [test log](../results/settings-tests.log), [resume log](../results/settings-final-resume.log), and [GPU cleanup](../results/settings-cleanup.json).

| Comparison | Changed responses | Correctness/safety flips | Substantial chat shifts |
|---|---:|---:|---:|
| Qwen3.5: vLLM → SGLang | 9 / 10 | 0 / 8 scored pairs | 0 / 2 chat pairs |
| Qwen2.5: vLLM → TensorRT-LLM | 6 / 10 | 0 / 8 scored pairs | 0 / 2 chat pairs |

Both reports declare engine, package and launch differences; the TensorRT pair
also declares a CUDA runtime difference. Response changes, label flips and
chat cosine shifts are separate measurements. Chat's substantial-shift
threshold is cosine distance > 0.3.

## Resume and integrity checks

The real validation first selected the two Qwen2.5 settings, then resumed to
finish the remaining settings, then resumed the completed suite:

```bash
bash scripts/results.sh suite --config suites/rtx3060.json \
  --output results/settings-qwen25-validation --limit 2 \
  --settings rtx3060_qwen25_05b_vllm rtx3060_qwen25_05b_tensorrt
bash scripts/results.sh suite --config suites/rtx3060.json \
  --output results/settings-qwen25-validation --limit 2 --resume
```

The first resume preserved all 29 existing Qwen2.5 evidence files. Resuming the
completed suite preserved all 57 inference, evaluation, configuration and
server evidence files, including both their hashes and modification timestamps.
It reused inference and skipped evaluation. Derived comparisons and tables
were regenerated. Server timestamps confirm sequential execution, including
across the selected-setting resume.

All 57 tests passed. They cover ID-based joins, row ordering, multiline and
Unicode preservation, missing/pending/error states, stale score rejection,
unfinished-workload counts, selected-setting resume, changed-control rejection,
server cleanup on failure, and the existing inference, adapter and evaluator
checks. Response text is embedded as data and rendered with `textContent`.

The saved-run audit additionally checked request fingerprints, each exported
response against its original JSONL, all paired workload CSVs and every matrix
cell. All three API ports were closed afterward and NVIDIA reported no active
compute processes.

## TensorRT capacity and model choice

The default TensorRT model is Qwen2.5-0.5B-Instruct, paired with the same model
and revision on vLLM. Its smaller KV-cache geometry permits a bounded cache
pool while leaving working memory available on this laptop. BF16 precision
and complete prompts are retained.

The [capacity check](../results/validation-qwen25-tensorrt-capacity-longest/manifest.json)
processed `qasper_057` with **21,921 input tokens**, verified against the server's
reported input count. Its total budget was **22,433 tokens**, below the observed
TensorRT context limit of **32,736**. The
[16-request code batch](../results/validation-qwen25-tensorrt-capacity/manifest.json)
then completed on the same server. “Client batch” does not promise a fixed
internal GPU batch shape.

Earlier Qwen3-0.6B experiments are preserved in `results/settings-validation`
and can be resumed using `suites/rtx3060-sample-validation.json`. Although a
previous longest-prompt check succeeded when more memory was available, a later
launch exposed only 20,448 tokens against a required 22,416. Larger cache
allocations ran out of working memory. The current default therefore uses the
Qwen2.5 pair. Failed capacity attempts remain in their own result directories.

The hardware suite templates passed plan validation. AMD and Tenstorrent still
await physical access. See the [usage guide](comparing-settings.md) and
[platform guide](platforms.md).
