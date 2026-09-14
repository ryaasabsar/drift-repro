from dataclasses import replace
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from driftbench_runner import a100, blackhole_p150b, mi210, preset
from driftbench_runner.common import ROOT, read_json
from driftbench_runner.safety import judge_environment
from driftbench_runner.suite import load_plan


@pytest.mark.parametrize('module,vendor,judge,venv', [
    (mi210, 'amd', 'cuda', '.venv-rocm-vllm'),
    (blackhole_p150b, 'tenstorrent', 'cpu', '.venv-tt-vllm'),
])
def test_presets_keep_comparable_inputs_and_pinned_models(module, vendor, judge, venv, tmp_path):
    reference = load_plan(a100.SUITE, tmp_path / 'reference')
    plan = load_plan(module.SUITE, tmp_path / 'candidate')
    config = plan['settings'][0]['config']
    assert plan['expected_records_per_setting'] == 2284
    assert plan['sources'] == reference['sources']
    for field in ('model', 'revision', 'generation', 'seed', 'prompt_format', 'chat_template_kwargs', 'batch_size', 'batch_size_by_workload'):
        assert config[field] == reference['settings'][0]['config'][field]
    assert config['hardware']['vendor'] == vendor
    assert venv in plan['settings'][0]['server_python']
    assert plan['evaluation'] == {**reference['evaluation'], 'judge_device': judge}
    assert 'HF_TOKEN' not in json.dumps(plan)


@pytest.mark.parametrize('script', ['run_mi210.sh', 'run_blackhole_p150b.sh'])
def test_shell_dry_run_from_other_directory(script, tmp_path):
    output = tmp_path / 'unused'
    result = subprocess.run(['bash', str(ROOT / 'scripts' / script), '--dry-run', '--limit', '2', '--output', str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['expected_records_per_setting'] == 10
    assert not output.exists()


def test_mi210_requires_rocm_and_correct_gpu(monkeypatch):
    properties = SimpleNamespace(name='AMD Instinct MI210', total_memory=64 * 1024**3)
    cuda = SimpleNamespace(is_available=lambda: True, get_device_properties=lambda _: properties)
    version = SimpleNamespace(hip=None, cuda='12.8')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda, version=version))
    with pytest.raises(RuntimeError, match='ROCm PyTorch'):
        mi210.check_hardware()
    version.hip = '7.0'
    assert mi210.check_hardware()['rocm_runtime'] == '7.0'
    properties.name = 'AMD Radeon RX 7900 XT'
    with pytest.raises(RuntimeError, match='requires an Instinct MI210'):
        mi210.check_hardware()


@pytest.mark.parametrize('device', [None, '1'])
def test_mi210_visibility_set_before_preflight_and_preserves_scheduler(device, monkeypatch, capsys):
    monkeypatch.setenv('HIP_VISIBLE_DEVICES', '3')
    monkeypatch.setenv('ROCR_VISIBLE_DEVICES', 'GPU-fixture')
    monkeypatch.setattr(preset, 'load_credentials', lambda: None)
    def check(target, plan, inference_only):
        assert os.environ['HIP_VISIBLE_DEVICES'] == (device or '3')
        assert os.environ['ROCR_VISIBLE_DEVICES'] == 'GPU-fixture'
        assert target is mi210.PRESET
        assert plan['device'] == device
        return {'checked': True}
    monkeypatch.setattr(preset, 'check_requirements', check)
    mi210.main(['--check', *(['--device', device] if device else [])])
    assert json.loads(capsys.readouterr().out)['checked']


def test_blackhole_requires_opt_in_before_credentials_or_hardware(monkeypatch):
    monkeypatch.setattr(preset, 'load_credentials', lambda: pytest.fail('Unexpected credentials access'))
    monkeypatch.setattr(preset, 'check_requirements', lambda *args: pytest.fail('Unexpected hardware access'))
    with pytest.raises(SystemExit) as error:
        blackhole_p150b.main(['--limit', '1'])
    assert error.value.code == 1


def test_blackhole_opt_in_runs_its_suite_and_is_not_sent_to_suite_cli(monkeypatch):
    monkeypatch.setattr(preset, 'load_credentials', lambda: None)
    monkeypatch.setattr(preset, 'check_requirements', lambda *args: {})
    executed = []
    monkeypatch.setattr(preset.os, 'execv', lambda binary, args: executed.append(args))
    blackhole_p150b.main(['--allow-experimental', '--inference-only', '--limit', '1'])
    assert str(blackhole_p150b.SUITE) in executed[0]
    assert '--inference-only' in executed[0]
    assert '--allow-experimental' not in executed[0]


def test_blackhole_hardware_check_rejects_wormhole_and_missing_tt_stack(monkeypatch):
    device = {'vendor_id': '0x1e52', 'device_id': '0x401e'}
    hardware = {'tenstorrent_device_nodes': ['/dev/tenstorrent/0'], 'pci_devices': [device], 'packages': {}}
    monkeypatch.setattr(blackhole_p150b, 'discover', lambda: hardware)
    with pytest.raises(RuntimeError, match='No Blackhole'):
        blackhole_p150b.check_hardware()
    device['device_id'] = '0xb140'
    with pytest.raises(RuntimeError, match='TT-Metal/TTNN'):
        blackhole_p150b.check_hardware()
    hardware['packages'] = {'ttnn': 'fixture', 'vllm': 'fixture'}
    assert blackhole_p150b.check_hardware()['board_verified'] is False


def test_preflight_uses_selected_profile_and_reports_kernel_limit(tmp_path, monkeypatch):
    plan = load_plan(mi210.SUITE, tmp_path, limit=1)
    target = replace(mi210.PRESET, check_hardware=lambda: {'gpu': 'MI210 fixture'})
    monkeypatch.setattr(preset, 'check_model_access', lambda *args: None)
    profiles = []
    def prepare(path, *args):
        profiles.append(path)
        return None, None, [{'input_ids': [1, 2, 3]}], None
    monkeypatch.setattr('driftbench_runner.inference.prepare', prepare)
    report = preset.check_requirements(target, plan, True)
    assert read_json(profiles[0])['hardware']['vendor'] == 'amd'
    assert report['required_context'] == 515
    assert report['kernel_execution_tested'] is False
    assert report['evaluation'] is None


def test_judge_provenance_records_rocm_without_changing_cuda_metadata():
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda _: 'MI210')
    torch = SimpleNamespace(__version__='fixture', cuda=cuda, version=SimpleNamespace(cuda=None, hip='7.0'))
    assert judge_environment(torch, 'cuda')['rocm_runtime'] == '7.0'
    torch.version.hip = None
    torch.version.cuda = '12.8'
    assert 'rocm_runtime' not in judge_environment(torch, 'cuda')
    assert 'rocm_runtime' not in judge_environment(torch, 'cpu')
