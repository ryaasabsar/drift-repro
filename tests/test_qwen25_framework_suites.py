import json
import subprocess

import pytest

from driftbench_runner.common import ROOT, read_json
from driftbench_runner.serving import launch_command
from driftbench_runner.suite import load_plan


@pytest.mark.parametrize('target,frameworks', [
    ('a100', ['vllm', 'sglang', 'tensorrt-llm']),
    ('mi210', ['vllm', 'sglang']),
])
def test_framework_suites_preserve_pinned_experiment_controls(tmp_path, target, frameworks):
    plan = load_plan(ROOT / f'suites/{target}-qwen25-7b-frameworks.json', tmp_path)
    reference = read_json(ROOT / 'configs/a100-qwen25-7b-vllm-http.json')
    assert plan['expected_records_per_setting'] == 2284
    assert [item['config']['backend'] for item in plan['settings']] == frameworks
    assert plan['evaluation']['judge_revision'] == read_json(ROOT / 'models.lock.json')['safety_llamaguard3_8b']['revision']
    ports, environments = set(), set()
    for item in plan['settings']:
        config = item['config']
        for key in ('model', 'revision', 'prompt_format', 'chat_template_kwargs', 'generation', 'seed', 'batch_size', 'batch_size_by_workload'):
            assert config[key] == reference[key]
        assert config['engine']['max_model_len'] == 32768
        assert config['engine']['dtype'] == 'bfloat16'
        assert config['engine']['tensor_parallel_size'] == 1
        assert config['hardware']['vendor'] == ('nvidia' if target == 'a100' else 'amd')
        command = launch_command(config)
        assert command[command.index('--revision') + 1] == reference['revision']
        assert config['server']['metadata_path'].startswith(str(tmp_path / 'settings' / item['id']))
        ports.add(config['server']['base_url'])
        environments.add(item['server_python'])
        if config['backend'] == 'sglang':
            assert 'max_mamba_cache_size' not in config['engine']
            assert '--disable-radix-cache' in command and '--disable-cuda-graph' in command
            if target == 'mi210':
                assert 'cuda_version' not in config['launch']
                assert '.venv-rocm-sglang' in item['server_python']
    assert len(ports) == len(environments) == len(frameworks)


def test_tensorrt_profile_uses_correct_model_alias_and_preserves_context_budget():
    config = read_json(ROOT / 'configs/a100-qwen25-7b-tensorrt-http.json')
    assert config['server']['model_name'] == 'Qwen2.5-7B-Instruct'
    assert config['engine']['fail_fast_on_attention_window_too_large'] is True
    assert config['engine']['llm_api_options']['kv_cache_config']['max_tokens'] >= config['engine']['max_model_len']
    command = launch_command(config)
    assert '--extra_llm_api_options' in command
    assert '--enable_chunked_prefill' in command
    assert command[command.index('--backend') + 1] == 'pytorch'


@pytest.mark.parametrize('name', ['a100-qwen25-7b-sglang', 'mi210-qwen25-7b-sglang', 'a100-qwen25-7b-tensorrt'])
def test_standalone_framework_can_be_smoke_tested_without_other_frameworks(tmp_path, name):
    result = subprocess.run(['bash', str(ROOT / 'scripts/results.sh'), 'infer', '--config',
        str(ROOT / 'suites' / (name + '.json')), '--run-id', 'smoke', '--output-root', str(tmp_path / 'runs'),
        '--limit', '2', '--dry-run'], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan['settings']) == 1 and plan['expected_records_per_setting'] == 10
    assert not (tmp_path / 'runs').exists()
