"""Verify the portable CLI and custom input contract outside a parent project."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from normbench.data import load_fixture, make_fixture


def test_cli_outside_checkout_without_frameworks(tmp_path):
    # A separate interpreter avoids cached imports masking a hidden dependency.
    guard = tmp_path / 'sitecustomize.py'
    guard.write_text('''import sys
class BlockFrameworks:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'driftbench_runner', 'vllm', 'torch', 'triton', 'ttnn'}:
            raise ImportError('Standalone client imported ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
''')
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'normbench', 'run',
                             '--backend', 'tenstorrent', '--dry-run'], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(tmp_path), str(project)])},
                            text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['protocol']['control'] == 'native'
    assert len(plan['protocol']['cases']) == 6
    assert Path(plan['fixture']).is_relative_to(project)
    assert not (tmp_path / 'results').exists()


def test_custom_inputs_need_no_capture_directory(tmp_path):
    x = np.arange(128, dtype=np.float32).reshape(1, 128) / 128
    path = tmp_path / 'tensors.npz'
    np.savez(path, x=x, w=np.ones(128, np.float32), z=np.zeros_like(x))
    folder = tmp_path / 'fixture'
    make_fixture(folder, path, eps=0.001)
    manifest, arrays = load_fixture(folder)
    assert manifest['focus_positions'] == []
    assert manifest['cases'][0]['shape'] == [1, 128]
    assert manifest['cases'][0]['eps'] == 0.001
    np.testing.assert_array_equal(arrays['captured/input/x'], x)
    np.testing.assert_array_equal(arrays['captured/reference/output'], np.zeros_like(x))


def test_custom_inputs_reject_implicit_bf16_rounding(tmp_path):
    path = tmp_path / 'tensors.npz'
    np.savez(path, x=np.full((1, 128), 1.001, np.float32),
             w=np.ones(128, np.float32), z=np.zeros((1, 128), np.float32))
    with pytest.raises(ValueError, match='quantize explicitly'):
        make_fixture(tmp_path / 'fixture', path)
