"""Check production wiring without allocating an accelerator or fetching weights."""
from collections import defaultdict
import os
import subprocess

import pytest

from driftbench_runner.common import ROOT, read_json
from driftbench_runner.serving import launch_command
from driftbench_runner.suite import load_plan


def test_complete_matrix_and_matching_experimental_controls(tmp_path):
    expected = {'a100': {'vllm', 'sglang'}, 'mi210': {'vllm', 'sglang'},
                'blackhole-p150b': {'vllm'}}
    controls = defaultdict(list)
    seen = set()
    assert {p.stem for p in (ROOT / 'suites').glob('*.json')} == set(expected)
    for host, backends in expected.items():
        plan = load_plan(ROOT / 'suites' / f'{host}.json', tmp_path / host)
        assert len(plan['settings']) == 3 * len(backends)
        assert plan['limit'] is None and plan['expected_records_per_setting'] == 2284
        assert plan['evaluation']['judge_model'] == 'meta-llama/Llama-Guard-3-8B'
        assert plan['evaluation']['judge_device'] == 'cuda'
        combinations = set()
        for item in plan['settings']:
            c = item['config']
            assert c['transport'] == 'http'
            assert c['backend'] in backends
            assert item['id'] not in seen
            seen.add(item['id'])
            combinations.add((c['model'], c['backend']))
            argv = launch_command(c)
            assert '--language-only' not in argv
            assert c['seed'] == 42 and c['batch_size'] == 1
            assert c['batch_size_by_workload'] == {'long_context': 1}
            assert c['generation'] == {
                'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'max_tokens': 512,
                'repetition_penalty': 1.0, 'presence_penalty': 0.0,
                'frequency_penalty': 0.0, 'min_p': 0.0,
            }
            assert c['launch']['env']['PYTHONHASHSEED'] == '42'
            if c['model'] == 'meta-llama/Llama-3.1-8B-Instruct':
                assert c['chat_template_kwargs']['date_string'] == '15 Sep 2026'
            if c['backend'] == 'sglang':
                assert '--enable-deterministic-inference' in argv
                assert argv[argv.index('--sampling-defaults') + 1] == 'openai'
                assert c['engine']['max_running_requests'] == 1
            else:
                assert c['engine']['max_num_seqs'] == 1
            from driftbench_runner.http_backend import HTTPBackend
            _, payload = HTTPBackend(c).payload({'input_ids': [1, 2, 3]})
            params = payload['sampling_params'] if c['backend'] == 'sglang' else payload
            seed_key = 'sampling_seed' if c['backend'] == 'sglang' else 'seed'
            budget_key = 'max_new_tokens' if c['backend'] == 'sglang' else 'max_tokens'
            assert params[seed_key] == 42 and params[budget_key] == 512
            for key, value in c['generation'].items():
                if key != 'max_tokens':
                    assert params[key] == value
            controls[c['model']].append({k:c[k] for k in (
                'revision', 'prompt_format', 'chat_template_kwargs', 'seed', 'generation',
                'batch_size', 'batch_size_by_workload')})
        assert len(combinations) == len(plan['settings'])
    assert len(seen) == 15
    assert set(controls) == {'Qwen/Qwen3.5-9B-Base', 'Qwen/Qwen2.5-7B-Instruct',
                             'meta-llama/Llama-3.1-8B-Instruct'}
    assert all(len(rows) == 5 and all(row == rows[0] for row in rows) for rows in controls.values())
    assert len(list((ROOT / 'configs').glob('*.json'))) == 15


@pytest.mark.parametrize('host', ['a100', 'mi210', 'blackhole_p150b'])
def test_host_entry_point_dry_run_is_inference_only(host, tmp_path):
    output = tmp_path / 'runs'
    result = subprocess.run([
        'bash', str(ROOT / 'scripts' / f'run_{host}.sh'), '--run-id', 'matrix-check',
        '--dry-run', '--output-root', str(output),
    ], cwd=tmp_path, env={**os.environ, 'DRIFTBENCH_PYTHON': str(ROOT / '.venv/bin/python')},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'expected_records_per_setting' in result.stdout
    assert not output.exists()
