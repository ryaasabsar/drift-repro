"""Check installer boundaries without downloading wheels or requiring an AMD GPU."""
import os
import shutil
import subprocess

import pytest

from driftbench_runner.common import ROOT, read_json
from driftbench_runner.software import check_runtime


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / 'workspace'
    (root / 'scripts').mkdir(parents=True)
    for name in ('install_rocm_vllm.sh', 'env.sh'):
        shutil.copy(ROOT / 'scripts' / name, root / 'scripts' / name)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    getconf = bindir / 'getconf'
    getconf.write_text('#!/bin/sh\nprintf "glibc %s\\n" "${TEST_GLIBC:-2.35}"\n')
    getconf.chmod(0o755)
    return root, dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'])


def invoke(workspace, *args):
    root, env = workspace
    return subprocess.run(['bash', str(root / 'scripts/install_rocm_vllm.sh'), *args],
                          env=env, capture_output=True, text=True, timeout=10)


def test_dry_run_is_nonmutating_and_targets_rocm(workspace):
    root, _ = workspace
    before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
    result = invoke(workspace, '--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--python 3.12.14' in result.stdout
    assert 'https://wheels.vllm.ai/rocm/0.17.1/rocm700' in result.stdout
    assert '.venv-rocm-vllm/bin/python' in result.stdout
    assert 'requirements.rocm-vllm.lock.txt' in result.stdout
    assert 'pip sync' in result.stdout
    assert 'install_cuda' not in result.stdout
    assert before == sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def test_unsupported_glibc_fails_before_installation(workspace):
    root, env = workspace
    env['TEST_GLIBC'] = '2.31'
    result = invoke(workspace)
    assert result.returncode == 1
    assert 'glibc >=2.35' in result.stderr
    assert not (root / '.tools').exists()


def test_symlink_target_refused(workspace):
    root, _ = workspace
    (root / '.venv-rocm-vllm').symlink_to(root / 'other-environment')
    result = invoke(workspace, '--dry-run')
    assert result.returncode == 1
    assert 'Refusing symlink/non-venv' in result.stderr


def test_check_gpu_does_not_install(workspace):
    result = invoke(workspace, '--check-gpu', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert 'Triton initialization' in result.stdout
    assert 'pip install' not in result.stdout
    assert 'pip sync' not in result.stdout
    assert 'ensure_uv' not in result.stdout


def test_rocm_contract_preserves_native_pair_and_shared_pins():
    pins = dict(line.split('==') for line in (ROOT / 'requirements.rocm-vllm.in').read_text().splitlines()
                if line and not line.startswith('#'))
    nvidia = dict(line.split('==') for line in (ROOT / 'requirements.lock.txt').read_text().splitlines()
                  if line and not line.startswith('#'))
    locked = dict(line.split('==') for line in (ROOT / 'requirements.rocm-vllm.lock.txt').read_text().splitlines()
                  if line and not line.startswith('#'))
    for name, version in pins.items():
        assert locked[name] == version
        if name not in ('torch', 'vllm'):
            assert nvidia[name] == version
    metadata = {'python': '3.12.14', 'packages': pins, 'rocm_runtime': '7.0.51831'}
    config = {'hardware': {'vendor': 'amd'}, 'backend': 'vllm'}
    assert check_runtime(config, metadata)['status'] == 'match'
    metadata['packages'] = {**pins, 'torch': '2.10.0+cu128'}
    metadata['rocm_runtime'] = None
    differences = check_runtime(config, metadata)['differences']
    assert 'torch' in differences and 'rocm_runtime' in differences
    assert read_json(ROOT / 'runtime-contracts.json')['serving']['nvidia/vllm']['packages']['torch'] == '2.10.0'
