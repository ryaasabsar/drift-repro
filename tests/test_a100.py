import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from driftbench_runner import a100
from driftbench_runner.common import ROOT, read_json
from driftbench_runner.suite import load_plan


def test_a100_plan_pins_models_and_keeps_five_workloads(tmp_path):
    plan = load_plan(a100.SUITE, tmp_path)
    config = plan['settings'][0]['config']
    lock = read_json(ROOT / 'models.lock.json')
    assert plan['expected_records_per_setting'] == 2284
    assert config['model'] == 'Qwen/Qwen2.5-7B-Instruct'
    assert config['revision'] == lock['inference_a100_qwen25_7b']['revision']
    assert plan['evaluation']['judge_model'] == 'meta-llama/Llama-Guard-3-8B'
    assert plan['evaluation']['judge_revision'] == lock['safety_llamaguard3_8b']['revision']
    assert plan['evaluation']['judge_device'] == 'cuda'
    assert config['engine']['tensor_parallel_size'] == 1
    assert config['batch_size_by_workload']['long_context'] == 1


def test_credential_file_is_data_and_not_forwarded_in_arguments(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'credentials.json'
    token = 'hf_fixture_$(touch should_not_exist)'
    path.write_text(json.dumps({'HF_TOKEN': token}))
    monkeypatch.delenv('HF_TOKEN', raising=False)
    a100.load_credentials(path)
    assert os.environ['HF_TOKEN'] == token
    args = a100.parser().parse_args(['--limit', '2', '--device', '1', '--resume'])
    assert token not in ' '.join(a100.suite_arguments(args))
    assert not Path('should_not_exist').exists()
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err


def test_empty_credentials_preserve_exported_token(tmp_path, monkeypatch):
    path = tmp_path / 'credentials.json'
    path.write_text('{"HF_TOKEN": ""}')
    monkeypatch.setenv('HF_TOKEN', 'hf_exported_fixture')
    a100.load_credentials(path)
    assert os.environ['HF_TOKEN'] == 'hf_exported_fixture'


def test_bad_credentials_do_not_disclose_contents(tmp_path):
    path = tmp_path / 'credentials.json'
    path.write_text('{"HF_TOKEN": "private_fixture_token"')
    with pytest.raises(ValueError) as error:
        a100.load_credentials(path)
    assert 'private_fixture_token' not in str(error.value)


def test_dry_run_needs_neither_credentials_nor_gpu(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('Dry run performed a real preflight')
    monkeypatch.setattr(a100, 'load_credentials', forbidden)
    monkeypatch.setattr(a100, 'check_requirements', forbidden)
    executed = []
    monkeypatch.setattr(a100.os, 'execv', lambda binary, args: executed.append(args))
    a100.main(['--dry-run', '--limit', '2', '--output', str(tmp_path / 'unused')])
    assert '--dry-run' in executed[0]
    assert not (tmp_path / 'unused').exists()


def test_hardware_guard_rejects_small_gpu_and_accepts_a100(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, get_device_properties=lambda _: SimpleNamespace(name='RTX 3060', total_memory=6*1024**3))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda='12.8')))
    with pytest.raises(RuntimeError, match='A100'):
        a100.check_hardware()
    cuda.get_device_properties = lambda _: SimpleNamespace(name='NVIDIA A100-SXM4-40GB', total_memory=40*1024**3)
    assert a100.check_hardware()['memory_gib'] == 40


def test_gated_access_fails_before_qwen_preparation(tmp_path, monkeypatch):
    plan = load_plan(a100.SUITE, tmp_path)
    monkeypatch.setattr(a100, 'check_hardware', lambda: {'gpu': 'A100'})
    models = []
    def denied(model, revision):
        models.append(model)
        raise RuntimeError('Access denied')
    monkeypatch.setattr(a100, 'check_model_access', denied)
    with pytest.raises(RuntimeError, match='Access denied'):
        a100.check_requirements(plan, False)
    assert models == ['meta-llama/Llama-Guard-3-8B']


def test_inference_only_skips_guard_and_code_worker(tmp_path, monkeypatch):
    plan = load_plan(a100.SUITE, tmp_path, limit=1)
    models = []
    monkeypatch.setattr(a100, 'check_hardware', lambda: {'gpu': 'A100'})
    monkeypatch.setattr(a100, 'check_model_access', lambda model, revision: models.append(model))
    monkeypatch.setattr('driftbench_runner.inference.prepare', lambda *args: (None, None, [{'input_ids': [1, 2]}], None))
    monkeypatch.setattr('driftbench_runner.evaluation.execute_code', lambda *args: pytest.fail('Code evaluator should be deferred'))
    report = a100.check_requirements(plan, True)
    assert models == ['Qwen/Qwen2.5-7B-Instruct']
    assert report['evaluation'] is None
    assert report['required_context'] == 514


def test_shell_wrapper_produces_json_from_other_directory(tmp_path):
    out = tmp_path / 'unused'
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_a100.sh'), '--dry-run', '--limit', '2', '--output', str(out)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['expected_records_per_setting'] == 10
    assert not out.exists()


def test_secret_location_ignored_and_private(tmp_path):
    assert '/.secrets/' in (ROOT / '.gitignore').read_text().splitlines()
    path = a100.ensure_credentials_file(tmp_path / '.secrets/huggingface.json')
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    path.write_text('{"HF_TOKEN": "hf_fixture_existing"}')
    a100.ensure_credentials_file(path)
    assert json.loads(path.read_text())['HF_TOKEN'] == 'hf_fixture_existing'
