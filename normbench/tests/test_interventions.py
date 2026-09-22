"""Causality controls, portable inputs, and effect accounting; no GPU required."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from normbench.common import digest, write_json
from normbench.cli import seal
from normbench.data import STAGES, logical_bits, reference
from normbench.interventions import (BUNDLE, EXPERIMENT, PRIMARY, effect_counts,
    load_bundle, compare, key, save_kernel, verify_artifacts, unaffected_stages)


def test_bundle_freezes_actual_fp32_operand_and_shared_targets():
    meta, data = load_bundle()
    assert meta['baseline_differing_elements'] == 3
    # The real device operand, not a recomputed ideal float64 reduction.
    np.testing.assert_array_equal(data['rsqrt_input'].view(np.uint32), data['nvidia/trace/add_epsilon'])
    np.testing.assert_array_equal(data['amd/trace/add_epsilon'], data['nvidia/trace/add_epsilon'])
    for op, arg in (('rsqrt', 'rsqrt_input'), ('sigmoid', 'z')):
        target = reference(op, [data[arg]], meta['case']['shape'][1], meta['case']['eps']).astype(np.float32)
        np.testing.assert_allclose(data['shared_' + op], target, rtol=1e-7, atol=0)
    assert data['x'].shape == (1552, 128)


def test_bundle_integrity(tmp_path):
    (tmp_path / 'bundle.json').write_bytes((BUNDLE / 'bundle.json').read_bytes())
    (tmp_path / 'inputs.npz').write_bytes(b'broken archive')
    with pytest.raises(ValueError, match='Changed bundle tensors'):
        load_bundle(tmp_path)


def test_disappearance_and_new_differences_count_separately():
    a = np.array([1, 2, 3, 4], np.uint16)
    before = np.array([2, 3, 3, 4], np.uint16)
    after = np.array([1, 3, 4, 4], np.uint16)
    r = effect_counts(a, before, a, after)
    assert r == dict(baseline_differing=2, differing=2, differing_percent=50.,
                     resolved=1, remaining=1, new=1, elements=4)


def test_unchanged_branch_control_includes_other_operation():
    assert 'rsqrt' in unaffected_stages(False, True)
    assert 'sigmoid' in unaffected_stages(True, False)
    assert 'gate' in unaffected_stages(True, False)
    assert unaffected_stages(True, True) == ['square', 'sum_squares', 'mean_square', 'add_epsilon']


@pytest.mark.parametrize('backend,assembly,binary', [('nvidia', 'ptx', 'cubin'), ('amd', 'amdgcn', 'hsaco')])
def test_export_actual_compiled_artifacts_and_reject_tampering(tmp_path, backend, assembly, binary):
    asm = dict(ttir='IR', ttgir='GPU IR', llir='llvm sqrt and fdiv', **{assembly: 'rsqrt instruction', binary: b'actual binary'})
    kernel = SimpleNamespace(asm=asm, metadata=SimpleNamespace(_asdict=lambda: {'target': backend}))
    m = save_kernel(kernel, tmp_path / 'kernels/original-native', {'num_warps': 1}, backend)
    result = {'kernel_artifacts': {'original-native': m}, 'source_files': {}}
    verify_artifacts(tmp_path, result)
    assert m['instruction_excerpts'][assembly] == ['rsqrt instruction']
    (tmp_path / f'kernels/original-native/kernel.{binary}').write_bytes(b'modified')
    with pytest.raises(ValueError, match='Changed compiled kernel'):
        verify_artifacts(tmp_path, result)


def test_missing_assembly_is_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match='Missing compiled ttir'):
        save_kernel(SimpleNamespace(asm={}), tmp_path / 'kernel', {}, 'nvidia')


def make_campaign(path, candidate=False, control_pass=True, restart_change=False):
    """Synthetic sealed results test reporting policy, not device execution."""
    protocol = {'experiment': EXPERIMENT, 'cases': ['captured'], 'precisions': ['float32'],
                'repeats': 2, 'stages': list(STAGES), 'bundle_fingerprint': 'test'}
    path.mkdir()
    write_json(path / 'campaign.json', {'status': 'complete', 'protocol': protocol, 'processes': 2})
    for i in (1, 2):
        out = path / f'process-{i:03d}'
        out.mkdir()
        records, arrays = {}, {}
        for variant in PRIMARY:
            value = np.array([[1, 2, 3, 4]], dtype=np.uint16) + 0x3f80
            if candidate:
                offsets = {'original': [1, 1, 0, 0], 'shared_sigmoid': [0, 1, 1, 0],
                           'shared_rsqrt': [1, 0, 0, 0], 'shared_both': [0, 0, 0, 0]}
                value += np.array([offsets[variant]], dtype=np.uint16)
            if restart_change and i == 2 and variant == 'shared_both':
                value[0, 3] += 1
            k = key(variant, native=True)
            arrays[k] = value
            records[k] = {'case': 'captured', 'precision': 'float32', 'mode': variant + '-native',
                          'stage': 'output', 'dtype': 'bfloat16', 'shape': [1, 4], 'repeat_bitwise_equal': True,
                          'repeat_sha256': [digest(value.tobytes())] * 2, 'operand_fingerprint': 'test',
                          'reference_errors': [{'differing_percent': 0., 'relative_l2_error': 0.}]}
        np.savez_compressed(out / 'outputs.npz', **arrays)
        controls = {'variants': {v: {'attribution_eligible': control_pass} for v in PRIMARY}}
        seal(out / 'result.json', {'status': 'complete', 'protocol': protocol, 'records': records,
             'environment': {'backend': 'amd' if candidate else 'nvidia'}, 'controls': controls,
             'arrays_sha256': digest((out / 'outputs.npz').read_bytes()), 'kernel_artifacts': {}, 'source_files': {}})
    return path


def test_comparison_tracks_new_differences_and_suppresses_failed_controls(tmp_path):
    left = make_campaign(tmp_path / 'left')
    right = make_campaign(tmp_path / 'right', candidate=True, control_pass=False)
    report = compare(left, right, tmp_path / 'comparison')
    rows = {r['variant']: r for r in report['effects'] if r['process'] == 1}
    assert rows['shared_sigmoid']['resolved'] == rows['shared_sigmoid']['remaining'] == rows['shared_sigmoid']['new'] == 1
    assert rows['shared_both']['differing'] == 0
    assert all(not r['attribution_eligible'] for r in report['effects'])
    assert (tmp_path / 'comparison/coordinates.csv').is_file()


def test_fresh_process_drift_invalidates_attribution(tmp_path):
    left = make_campaign(tmp_path / 'left')
    right = make_campaign(tmp_path / 'right', candidate=True, restart_change=True)
    report = compare(left, right, tmp_path / 'comparison')
    assert all(not r['attribution_eligible'] for r in report['effects'])


def test_reject_different_intervention_protocols(tmp_path):
    left = make_campaign(tmp_path / 'left')
    right = make_campaign(tmp_path / 'right')
    p = right / 'campaign.json'
    meta = json.loads(p.read_text()); meta['protocol']['bundle_fingerprint'] = 'different'
    write_json(p, meta)
    with pytest.raises(ValueError, match='Different intervention protocols'):
        compare(left, right, tmp_path / 'comparison')
    assert not (tmp_path / 'comparison').exists()


def test_intervention_dry_run_needs_no_accelerator_imports(tmp_path):
    (tmp_path / 'sitecustomize.py').write_text('''import sys
class BlockAccelerators:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'triton', 'vllm', 'driftbench_runner'}:
            raise ImportError('Client imported ' + name)
sys.meta_path.insert(0, BlockAccelerators())
''')
    project = Path(__file__).resolve().parents[1]
    output = subprocess.run([sys.executable, '-m', 'normbench', 'intervene', '--backend', 'amd', '--dry-run'],
                            cwd=tmp_path, env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(tmp_path), str(project)])},
                            text=True, capture_output=True, check=True)
    plan = json.loads(output.stdout)
    assert len(plan['protocol']['variants']) == 8
    assert plan['protocol']['bundle_fingerprint']
    assert not (tmp_path / 'results').exists()


def source_campaign(path, backend, bundle, data, bad_operand=False, bad_control=False):
    path.mkdir()
    protocol = bundle['source_protocol']
    write_json(path / 'campaign.json', {'status': 'complete', 'processes': 2, 'protocol': protocol})
    arrays, records = {}, {}
    for stage in STAGES:
        k = f'captured/float32/instrumented/{stage}'
        arrays[k] = data[f'{backend}/trace/{stage}'].copy()
        if bad_operand and stage == 'add_epsilon':
            arrays[k][0, 0] += 1
    arrays['captured/float32/native/output'] = data[f'{backend}/native'].copy()
    for k, value in arrays.items():
        records[k] = {'dtype': 'bfloat16' if k.endswith('/output') else 'float32',
                      'shape': list(value.shape), 'repeat_sha256': [digest(value.tobytes())] * 3,
                      'repeat_bitwise_equal': True}
    for i in (1, 2):
        out = path / f'process-{i:03d}'; out.mkdir()
        np.savez_compressed(out / 'outputs.npz', **arrays)
        seal(out / 'result.json', {'status': 'complete', 'protocol': protocol, 'records': records,
            'environment': {'backend': backend}, 'controls': {'captured': {'native_repeatable': True,
                'native_before_after_equal': True, 'instrumented_matches_native': not bad_control}},
            'native_control': {'implementation': 'NormBench fused Triton kernel, snapshots disabled'},
            'arrays_sha256': digest((out / 'outputs.npz').read_bytes())})
    return path


@pytest.mark.parametrize('condition,error', [('operand', 'Actual rsqrt operands differ'),
                                            ('control', 'Source observer controls failed')])
def test_preparation_rejects_uncontrolled_source_experiments(tmp_path, condition, error):
    from normbench.interventions import prepare_bundle
    bundle, data = load_bundle()
    a = source_campaign(tmp_path / 'nvidia', 'nvidia', bundle, data)
    b = source_campaign(tmp_path / 'amd', 'amd', bundle, data,
                        bad_operand=condition == 'operand', bad_control=condition == 'control')
    with pytest.raises(ValueError, match=error):
        prepare_bundle(a, b, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
