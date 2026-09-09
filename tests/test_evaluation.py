import pytest

from driftbench_runner.common import keyed, load_workloads
from driftbench_runner.evaluation import code_candidate, execute_code, extract_number, qa_f1, wilson
from driftbench_runner.safety import parse_judgment


@pytest.mark.parametrize("text,expected", [
    ("The cost was 2,400 and the final answer is $1,200.00.", "1.2E+3"),
    ("Steps: 10, 4. #### -6", "-6"),
    (r"The answer is \boxed{0.5}. Step 3 confirms this.", "0.5"),
    ("Final answer: 1/2", "0.5"),
    ("<think>10*20=200</think>No answer available", None),
    ("No numeric answer.", None),
])
def test_numeric_answer(text, expected):
    assert extract_number(text) == expected


def test_longbench_f1_and_threshold():
    assert qa_f1("The red fox.", "red fox") == 1
    assert qa_f1("red fox", "red dog") == 0.5
    assert qa_f1("yes, clearly", "yes") == 0
    assert qa_f1("unanswerable", "unanswerable") == 1


def test_wilson_not_zero_uncertainty_for_no_flips():
    low, high = wilson(0, 10)
    assert low < 1e-10
    assert 0.27 < high < 0.28
    assert wilson(0, 0) is None


def test_published_dataset_ids_unique_and_complete():
    rows, sources = load_workloads(["code", "math", "safety", "chat", "long_context"])
    assert len(rows) == 2284
    assert sources["chat"]["selected"] == 1000
    assert len(keyed(rows)) == len(rows)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        keyed([{"workload": "math", "prompt_id": "x"}] * 2)


def test_code_extraction_full_function_and_continuation():
    prompt = "from typing import List\n\ndef f(x: List[int]):\n    '''sum'''\n"
    assert "from typing import List" in code_candidate(prompt, "```python\ndef f(x):\n    return sum(x)\n```", "f")
    assert code_candidate(prompt, "    return sum(x)\n", "f") == prompt + "    return sum(x)\n"


def test_guard_parsers_preserve_severity():
    assert parse_judgment("Safety: Safe\nCategories: None\nRefusal: Yes", "qwen3guard") == ("safe", "safe")
    assert parse_judgment("Safety: Controversial\nCategories: Unethical Acts", "qwen3guard") == ("unsafe", "controversial")
    assert parse_judgment("unsafe\nS1", "llamaguard3") == ("unsafe", "unsafe")
    with pytest.raises(ValueError):
        parse_judgment("I cannot classify", "qwen3guard")


def test_real_humaneval_execution():
    rows, _ = load_workloads(["code"], 1)
    correct = execute_code(rows[0], rows[0]["canonical_solution"])
    if correct["status"] == "pending":
        pytest.skip(correct["reason"])
    assert correct["correct"] is True
    incorrect = execute_code(rows[0], "    return False\n")
    assert incorrect["correct"] is False


def test_humaneval_cannot_read_workspace():
    row = {"prompt": "def f():\n", "entry_point": "f", "test_cases": "def check(f):\n    assert f() is True"}
    from driftbench_runner.common import ROOT
    output = f"    import os\n    return not os.path.exists({str(ROOT)!r})\n"
    result = execute_code(row, output)
    if result["status"] == "pending":
        pytest.skip(result["reason"])
    assert result["correct"] is True


def test_unclosed_fence_preserves_prompt_helpers():
    row = {"prompt": "def helper(x):\n    return x + 1\n\ndef f(x):\n    '''Use the supplied helper.'''\n",
           "entry_point": "f", "test_cases": "def check(f):\n    assert f(4) == 5"}
    output = "Here is the function:\n```python\ndef f(x):\n    return helper(x)\n"
    result = execute_code(row, output)
    if result["status"] == "pending":
        pytest.skip(result["reason"])
    assert result["correct"] is True


def test_generated_main_examples_are_not_the_test_suite():
    row = {"prompt": "def f(x):\n    '''Increment.'''\n", "entry_point": "f",
           "test_cases": "def check(f):\n    assert f(4) == 5"}
    output = "```python\nfrom __future__ import annotations\ndef f(x):\n    return x + 1\nif __name__ == '__main__':\n    raise RuntimeError('Example only')\n```"
    result = execute_code(row, output)
    if result["status"] == "pending":
        pytest.skip(result["reason"])
    assert result["correct"] is True


def test_worker_random_seed_and_hash_seed():
    import os
    import subprocess
    import sys
    expected_hash = int(subprocess.check_output(
        [sys.executable, "-s", "-S", "-c", "print(hash('driftbench'))"],
        env={"PATH": os.defpath, "PYTHONHASHSEED": "42"}, text=True))
    row = {"prompt": "def f():\n    '''Return test values.'''\n", "entry_point": "f",
           "test_cases": f"def check(f):\n    assert f() == (0.6394267984578837, {expected_hash})"}
    result = execute_code(row, "    import random\n    return random.random(), hash('driftbench')\n")
    if result["status"] == "pending":
        pytest.skip(result["reason"])
    assert result["correct"] is True
