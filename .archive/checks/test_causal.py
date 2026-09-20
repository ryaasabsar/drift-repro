"""Integrity, localization, and captured-operand replay contracts (CPU harness)."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from driftbench_runner import causal
from driftbench_runner.causal_capture import make_mode
from driftbench_runner.causal_ops import (Archive, decode, logical_identity, metrics,
                                         reference_op, replay, validate_arrays, gdn_reference)
from driftbench_runner.common import digest, read_json, write_json


def save_trace(path, events, archive, **protocol_changes):
    path.mkdir()
    protocol = {'model': 'fixture', 'revision': 'pinned', 'dtype': 'bfloat16', 'seed': 42,
                'engine': {'enforce_eager': True}, 'case': {'input_ids': [1, 2]},
                'sampling': {'max_tokens': 1}, 'implementation_sha256': 'same',
                'weights_sha256': 'same', **protocol_changes}
    meta = {'schema_version': 1, 'status': 'complete', 'protocol': protocol,
            'execution': 'CPU test harness', 'target_module': 'layers.0',
            'environment': {'device': 'CPU'}, 'events': events,
            'tensors_sha256': archive.save(path / 'tensors.npz')}
    meta['fingerprint'] = digest(meta)
    write_json(path / 'trace.json', meta)
    return path


def mm_trace(path, perturb_output=False, perturb_input=False):
    archive = Archive()
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    if perturb_input:
        x[0, 0] = 2
    w = torch.tensor([[2., -1.], [1., 2.]], dtype=torch.bfloat16).t()
    y = x @ w
    if perturb_output:
        y[0, 0] += 1
    event = {'event': 0, 'kind': 'operation', 'name': 'aten.mm.default', 'module': 'layers.0',
             'schema': 'fixture', 'replayable': True, 'args': archive.encode((x, w)),
             'kwargs': archive.encode({}), 'outputs': archive.encode(y)}
    return save_trace(path, [event], archive)


def test_bf16_bits_stride_and_offset_roundtrip():
    x = torch.arange(35, dtype=torch.bfloat16).reshape(5, 7)[1:4, 1:6].t()
    archive = Archive()
    descriptor = archive.encode(x)
    y = decode(descriptor, archive.arrays)
    assert torch.equal(x.view(torch.int16), y.view(torch.int16))
    assert y.stride() == x.stride() and y.storage_offset() == x.storage_offset()
    validate_arrays(descriptor, archive.arrays)
    archive.arrays[descriptor['tensor']][0] ^= 1
    with pytest.raises(ValueError, match='Tensor changed'):
        validate_arrays(descriptor, archive.arrays)


def test_snapshots_survive_inplace_mutation():
    archive = Archive()
    x = torch.tensor([1., 2.])
    before = archive.encode(x)
    x.add_(10)
    assert decode(before, archive.arrays).tolist() == [1., 2.]


def test_localizes_same_inputs_and_rejects_upstream_claim(tmp_path):
    a = mm_trace(tmp_path / 'a')
    b = mm_trace(tmp_path / 'b', perturb_output=True)
    report = causal.compare(a, b, tmp_path / 'comparison')
    assert report['first_different_output']['event'] == 0
    assert report['operation_candidates'][0]['inputs_identical'] is True
    c = mm_trace(tmp_path / 'c', perturb_input=True)
    report = causal.compare(a, c, tmp_path / 'upstream')
    assert report['operation_candidates'][0]['inputs_identical'] is False


def test_export_move_replay_without_original_trace(tmp_path):
    import shutil
    trace = mm_trace(tmp_path / 'trace')
    causal.export_operation(trace, 0, tmp_path / 'export')
    moved = tmp_path / 'moved'
    (tmp_path / 'export').rename(moved)
    shutil.rmtree(trace)
    result = replay(moved, 'cpu', tmp_path / 'replay', repeats=2)
    assert result['repeat_bitwise_equal'] and result['all_within_tolerance']
    assert all(x['bitwise_equal_to_capture'] for r in result['repeats'] for x in r)
    replay(moved, 'cpu', tmp_path / 'replay2', repeats=2)
    compared = causal.compare_replays(tmp_path / 'replay', tmp_path / 'replay2', tmp_path / 'paired.json')
    assert all(r['bitwise_equal'] for r in compared['outputs'])
    meta = read_json(tmp_path / 'replay2/replay.json')
    meta['operation_fingerprint'] = 'different inputs'
    write_json(tmp_path / 'replay2/replay.json', meta)
    with pytest.raises(ValueError, match='operation_fingerprint'):
        causal.compare_replays(tmp_path / 'replay', tmp_path / 'replay2', tmp_path / 'reject-paired.json')


def test_rejects_mismatched_provenance_and_tampered_archives(tmp_path):
    a = mm_trace(tmp_path / 'a')
    b = mm_trace(tmp_path / 'b')
    meta = read_json(b / 'trace.json')
    meta['protocol']['case']['input_ids'] = [2, 1]
    meta['fingerprint'] = digest({k: v for k, v in meta.items() if k != 'fingerprint'})
    write_json(b / 'trace.json', meta)
    with pytest.raises(ValueError, match='Incomparable captures: case'):
        causal.compare(a, b, tmp_path / 'reject')
    with (a / 'tensors.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Trace arrays changed'):
        causal.export_operation(a, 0, tmp_path / 'reject-export')


def test_opaque_custom_kernel_is_not_fabricated_as_matmul(tmp_path):
    trace = mm_trace(tmp_path / 'trace')
    meta = read_json(trace / 'trace.json')
    meta['events'][0].update(name='vllm.gdn_attention_core.default', replayable=False)
    meta['fingerprint'] = digest({k: v for k, v in meta.items() if k != 'fingerprint'})
    write_json(trace / 'trace.json', meta)
    with pytest.raises(ValueError, match='dedicated replay/reference adapter'):
        causal.export_operation(trace, 0, tmp_path / 'reject')


def test_dispatch_records_real_operands_and_alias_limitation():
    class Recorder:
        busy = False
        target = 'fixture'
        def __init__(self):
            self.archive, self.events = Archive(), []
        def add(self, event):
            self.events.append(event)
    r = Recorder()
    x, w = torch.ones(2, 3), torch.ones(3, 2)
    with make_mode(r):
        y = x @ w
    assert len(r.events) == 1 and r.events[0]['name'] == 'aten.mm.default'
    assert r.events[0]['replayable']
    assert torch.equal(decode(r.events[0]['outputs'], r.archive.arrays), y)
    r = Recorder()
    with make_mode(r):
        z = x + x
    assert r.events[0]['aliased_inputs'] and not r.events[0]['replayable']


def test_numerical_reference_detects_error_and_nonfinite():
    x = torch.tensor([[1., 2.]], dtype=torch.float64)
    w = torch.tensor([1., 1.], dtype=torch.float64)
    got = reference_op('_C.rms_norm.default', (None, x, w, 0.), {})
    assert torch.allclose(got, x / np.sqrt(2.5))
    bad = metrics(np.array([1., np.nan]), np.array([1., 2.]))
    assert bad['nonfinite_pairs'] == 1 and not bad['within_tolerance']
    assert not metrics([1.1], [1.], atol=.01)['within_tolerance']


def test_byte_limit_fails_instead_of_silently_dropping_tensors():
    with pytest.raises(ValueError, match='byte limit exceeded'):
        Archive(max_bytes=4).encode(torch.ones(2))


def test_gated_delta_reference_nonzero_state_and_variable_lengths():
    # Scalar, manually evaluated recurrence with half decay and half update.
    # First sequence: 4 -> 2 + .5*(6-2) = 4 -> 2 + .5*(10-2) = 6.
    # Second sequence starts independently: 8 -> 4 + .5*(2-4) = 3.
    q = torch.ones(1, 3, 1, 1, dtype=torch.float64)
    v = torch.tensor([6., 10., 2.], dtype=torch.float64).reshape(1, 3, 1, 1)
    g = torch.full((1, 3, 1), -np.log(2.), dtype=torch.float64)
    beta = torch.full_like(g, .5)
    state = torch.tensor([4., 8.], dtype=torch.float64).reshape(2, 1, 1, 1)
    out, final = gdn_reference(q, q, v, g, beta, state, True, torch.tensor([0, 2, 3]), False)
    assert torch.allclose(out.flatten(), torch.tensor([4., 6., 3.], dtype=torch.float64))
    assert torch.allclose(final.flatten(), torch.tensor([6., 3.], dtype=torch.float64))
    # Reference must not mutate the captured initial state.
    assert state.flatten().tolist() == [4., 8.]


def test_gated_delta_reference_grouped_heads_normalization():
    q = torch.tensor([[[[2., 0.]]]], dtype=torch.float64)
    v = torch.tensor([[[[3.], [5.]]]], dtype=torch.float64)
    g = torch.zeros(1, 1, 2, dtype=torch.float64)
    beta = torch.ones_like(g)
    out, final = gdn_reference(q, q, v, g, beta, output_final_state=True)
    expected = v * (4. / (4. + 1e-6)) / np.sqrt(2.)
    assert torch.allclose(out, expected, rtol=1e-12, atol=1e-12)
    assert final.shape == (1, 2, 1, 2)
