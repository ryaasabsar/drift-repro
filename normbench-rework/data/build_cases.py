"""Rebuild data/cases.jsonl: all 164 HumanEval + the full GSM8K test split (1319).

Prompts are rendered exactly like DriftBench (Llama chat template, date_string
"15 Sep 2026") and tokenized with the pinned tokenizer. The script refuses to
write unless it reproduces the token IDs already used in earlier runs.

    python build_cases.py GSM8K_TEST.jsonl HUMANEVAL_PROMPTS.jsonl OLD_RUN_DIR
      GSM8K_TEST.jsonl       openai/gsm8k main/test as JSONL ({question, answer})
      HUMANEVAL_PROMPTS.jsonl  DriftBench datasets/humaneval_prompts.jsonl
      OLD_RUN_DIR            an earlier Llama-3.2-1B run folder with code.jsonl/math.jsonl
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL, REVISION = "meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"
gsm8k_path, humaneval_path, old_run = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def input_ids(prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, date_string="15 Sep 2026")
    return tok.encode(text, add_special_tokens=False)


rows = []
for r in map(json.loads, open(humaneval_path)):
    rows.append({"workload": "code", "prompt_id": r["prompt_id"], "prompt": r["prompt"],
                 "input_ids": input_ids(r["prompt"]), "entry_point": r["entry_point"],
                 "test_cases": r["test_cases"]})
for i, r in enumerate(map(json.loads, open(gsm8k_path))):
    rows.append({"workload": "math", "prompt_id": f"gsm8k_test_{i:04d}", "prompt": r["question"],
                 "input_ids": input_ids(r["question"]), "answer": r["answer"].rsplit("####", 1)[-1].strip()})

# Every prompt seen in the earlier run must tokenize to exactly the same IDs.
old = {}
for w in ("code", "math"):
    for r in map(json.loads, open(Path(old_run) / f"{w}.jsonl")):
        old[r["prompt"]] = r["input_token_ids"]
checked = [r for r in rows if r["prompt"] in old]
bad = [r["prompt_id"] for r in checked if r["input_ids"] != old[r["prompt"]]]
if bad or len(checked) < 600:
    sys.exit(f"tokenization check failed: {len(checked)} checked, mismatches {bad[:5]}")
with open(Path(__file__).with_name("cases.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} cases; {len(checked)} matched earlier token IDs exactly")
