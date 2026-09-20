"""Correctness contracts for model-free normalization experiments."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from normbench.common import read_json, write_json
from normbench.data import (FIXTURE, STAGES, PARENTS, bf16_bits,
    bf16_values, compare_values, load_fixture, reference)
from normbench import cli as normbench


def test_bf16_reference_avoids_double_rounding():
    # Actual disputed value: rounding to FP32 first lands on the BF16 midpoint.
    value = np.array([-0.0725097690879466], np.float64)
    assert bf16_values(bf16_bits(value))[0] == -0.07275390625
    assert bf16_values(bf16_bits(value.astype(np.float32)))[0] == -0.072265625


def test_bf16_ties_even_and_negative_zero():
    lo, hi = 1., 1.0078125
    mid = (lo + hi) / 2
    x = np.array([mid, np.nextafter(mid, np.inf), -mid, 0., -0.])
    np.testing.assert_array_equal(bf16_values(bf16_bits(x)), [lo, hi, -lo, 0., -0.])
    assert bf16_bits(x)[-1] == 0x8000


def test_ulp_signed_values_and_zero():
    a = bf16_values(np.array([0xbf80, 0x3f80, 0x8000], np.uint16))
    b = bf16_values(np.array([0xbf81, 0x3f81, 0x0000], np.uint16))
    stats = compare_values(a, b, 'bfloat16')
    assert stats['ulp_max'] == 1 and stats['differing_elements'] == 3
    assert not stats['bitwise_equal']


def test_nonfinite_outputs_not_reported_as_finite_success():
    result = compare_values(np.array([np.inf, np.nan], np.float32), np.ones(2), 'float32')
    assert result['nonfinite_pairs'] == 2 and result['ulp_max'] is None
    assert result['within_tolerance'] is False


def test_fixture_original_bits_and_references():
    meta, arrays = load_fixture()
    assert meta['focus_positions'] == [[761, 11], [1330, 36], [1385, 8]]
    for case in meta['cases']:
        if case['kind'] == 'cast':
            continue
        inputs = {k: arrays[f'{case["name"]}/input/{k}'] for k in ('x', 'w', 'z')}
        for value in inputs.values():
            np.testing.assert_array_equal(value, bf16_values(bf16_bits(value)))
        for stage in STAGES:
            inputs[stage] = reference(stage, [inputs[p] for p in PARENTS[stage]], case['shape'][-1], case['eps'])
            np.testing.assert_allclose(inputs[stage], arrays[f'{case["name"]}/reference/{stage}'], rtol=1e-14, atol=0)


def test_isolated_operands_are_independent_of_pipeline():
    _, data = load_fixture()
    key = 'captured/isolated/float32/rsqrt/arg0'
    original = data[key].copy()
    data['captured/reference/add_epsilon'] = np.zeros_like(original)
    np.testing.assert_array_equal(data[key], original)


def test_fixture_integrity_rejects_changed_archive(tmp_path):
    import shutil
    shutil.copy(FIXTURE / 'fixture.json', tmp_path)
    (tmp_path / 'cases.npz').write_bytes(b'wrong input file')
    with pytest.raises(ValueError, match='Changed fixture tensors'):
        load_fixture(tmp_path)


def execute_worker(path, monkeypatch=None, perturbed=False):
    pytest.importorskip('torch')
    path.mkdir()
    meta, _ = load_fixture()
    if perturbed:
        import normbench.backends as backends
        base = backends.TorchBackend
        class Perturbed(base):
            def operation(self, stage, args, width, eps):
                value = super().operation(stage, args, width, eps)
                if stage == 'rsqrt':
                    value = value * 1.01
                return value
        monkeypatch.setattr(backends, 'TorchBackend', Perturbed)
    protocol = {'fixture_fingerprint': meta['fingerprint'], 'implementation_sha256': normbench.implementation_hash(),
                'cases': ['synthetic-scale-1', 'bf16-midpoints'], 'precisions': ['float32'],
                'repeats': 2, 'stages': list(STAGES)}
    plan = {'output': str(path), 'fixture': str(FIXTURE), 'protocol': protocol, 'backend': 'cpu', 'tt_device_id': 0}
    write_json(path / 'plan.json', plan)
    normbench.worker(path / 'plan.json')
    return path


def test_real_cpu_worker_repeatability_and_stage_localization(tmp_path, monkeypatch):
    left = execute_worker(tmp_path / 'left')
    same = execute_worker(tmp_path / 'same')
    compared = normbench.compare_workers(left, same)
    assert all(r['bitwise_equal'] for r in compared['rows'])
    assert not compared['first_differing_stages']
    changed = execute_worker(tmp_path / 'changed', monkeypatch, perturbed=True)
    compared = normbench.compare_workers(left, changed)
    assert compared['first_differing_stages'][0]['stage'] == 'rsqrt'
    assert {r['stage'] for r in compared['isolated_differences']} == {'rsqrt'}
    meta, _ = normbench.load_result(left)
    assert all(r['repeat_bitwise_equal'] for r in meta['records'].values())
    assert meta['instrumentation']['status'] == 'unavailable'


def test_result_integrity_and_protocol_mismatch(tmp_path):
    left = execute_worker(tmp_path / 'left')
    right = execute_worker(tmp_path / 'right')
    report = read_json(right / 'result.json')
    report['protocol']['fixture_fingerprint'] = 'different'
    normbench.seal(right / 'result.json', report)
    with pytest.raises(ValueError, match='Different fixtures'):
        normbench.compare_workers(left, right)
    with (left / 'outputs.npz').open('ab') as stream:
        stream.write(b'tampering')
    with pytest.raises(ValueError, match='Changed output tensors'):
        normbench.load_result(left)


def test_rejects_silent_precision_change(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import normbench.backends as backends
    base = backends.TorchBackend
    class Downcast(base):
        def operation(self, stage, args, width, eps):
            return super().operation(stage, args, width, eps).to(torch.bfloat16)
    monkeypatch.setattr(backends, 'TorchBackend', Downcast)
    with pytest.raises(ValueError, match='no silent downcast'):
        execute_worker(tmp_path / 'failed')
    assert read_json(tmp_path / 'failed/result.json')['status'] == 'failed'


def test_instrumentation_failure_is_not_treated_as_native_evidence(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import normbench.backends as backends
    from normbench.data import SCALAR_STAGES
    base = backends.TorchBackend
    class UnfaithfulObserver(base):
        # Deliberately artificial operators test the observer gate, not GPU math.
        def native(self, inputs, eps):
            return torch.zeros(inputs['x'].shape, dtype=torch.bfloat16)
        def instrument(self, inputs, eps):
            m, n = inputs['x'].shape
            return {s: torch.ones((m, 1) if s in SCALAR_STAGES else (m, n),
                    dtype=torch.bfloat16 if s == 'output' else torch.float32) for s in STAGES}
    monkeypatch.setattr(backends, 'TorchBackend', UnfaithfulObserver)
    left = execute_worker(tmp_path / 'left')
    right = execute_worker(tmp_path / 'right')
    meta, _ = normbench.load_result(left)
    assert meta['controls']['synthetic-scale-1']['instrumented_matches_native'] is False
    comparison = normbench.compare_workers(left, right)
    assert all(r['instrumentation_controls_pass'] is False for r in comparison['rows'] if r['mode'] == 'instrumented')


def test_ttnn_dispatch_preserves_device_operations_and_accuracy_flags():
    from normbench.backends import TTBackend
    calls = []
    class Api:
        bfloat16 = 'bf16'
        def __getattr__(self, name):
            def operation(*args, **kwargs):
                calls.append((name, args, kwargs))
                return object()
            return operation
    adapter = TTBackend.__new__(TTBackend)
    adapter.tt = Api()
    adapter.compute_config = object()
    operand = object()
    adapter.operation('sum_squares', [operand], 128, 1e-6)
    adapter.operation('rsqrt', [operand], 128, 1e-6)
    adapter.operation('sigmoid', [operand], 128, 1e-6)
    adapter.operation('output', [operand], 128, 1e-6)
    assert [c[0] for c in calls] == ['sum', 'rsqrt', 'sigmoid_accurate', 'typecast']
    assert all(c[1][0] is operand for c in calls)
    assert calls[0][2]['compute_kernel_config'] is adapter.compute_config
    assert calls[1][2]['fast_and_approximate_mode'] is False
    assert calls[2][2]['fast_and_approximate_mode'] is False


def test_dry_run_needs_no_device(tmp_path):
    result = normbench.run(SimpleNamespace(fixture=FIXTURE, repeats=2, processes=2, timeout=10,
        cases=['captured'], precisions=['float32'], backend='tenstorrent', tt_device_id=0,
        output=tmp_path / 'dry', dry_run=True, control='native'))
    assert result['backend'] == 'tenstorrent' and not (tmp_path / 'dry').exists()
