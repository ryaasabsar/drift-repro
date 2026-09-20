"""Check the ROCm/Ray mask adaptation preserves scheduler restrictions."""
import os

import pytest

from driftbench_runner.serving import configure_serving_environment


@pytest.mark.parametrize('vendor,backend,masks,expected_hip', [
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2'}, '0'),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': 'GPU-deadbeef'}, '0'),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2', 'CUDA_VISIBLE_DEVICES': '0'}, '0'),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '0,2'}, None),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': ''}, None),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '-1'}, None),
    ('amd', 'vllm', {}, None),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2', 'CUDA_VISIBLE_DEVICES': '2'}, None),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2', 'CUDA_VISIBLE_DEVICES': ''}, None),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2', 'GPU_DEVICE_ORDINAL': '1'}, None),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2', 'HIP_VISIBLE_DEVICES': ''}, ''),
    ('amd', 'vllm', {'ROCR_VISIBLE_DEVICES': '2', 'HIP_VISIBLE_DEVICES': '1'}, '1'),
    ('nvidia', 'vllm', {'ROCR_VISIBLE_DEVICES': '2'}, None),
    ('tenstorrent', 'vllm', {'ROCR_VISIBLE_DEVICES': '2'}, None),
    ('amd', 'sglang', {'ROCR_VISIBLE_DEVICES': '2'}, None),
])
def test_single_rocr_device_ray_compatibility(monkeypatch, vendor, backend, masks, expected_hip):
    # The real function mutates os.environ, so isolate the entire mapping.
    monkeypatch.setattr(os, 'environ', {'PATH': '/usr/bin', **masks})
    configure_serving_environment({'hardware': {'vendor': vendor}, 'backend': backend})
    assert os.environ.get('HIP_VISIBLE_DEVICES') == expected_hip
    assert all(os.environ[key] == value for key, value in masks.items())


def test_explicit_launch_mask_is_preserved(monkeypatch):
    monkeypatch.setattr(os, 'environ', {'PATH': '/usr/bin', 'ROCR_VISIBLE_DEVICES': '2'})
    configure_serving_environment({'hardware': {'vendor': 'amd'}, 'backend': 'vllm',
                                   'launch': {'env': {'HIP_VISIBLE_DEVICES': ''}}})
    assert os.environ['HIP_VISIBLE_DEVICES'] == ''
    assert os.environ['ROCR_VISIBLE_DEVICES'] == '2'
