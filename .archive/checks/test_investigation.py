from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from driftbench_runner import investigation as inv
from driftbench_runner import investigation_probes as probes
from driftbench_runner.common import append_jsonl, digest, keyed, read_json, read_jsonl, write_json
from driftbench_runner.serving import serving_identity
from test_stages import make_run


@pytest.fixture
def bundle(tmp_path):
    runs = {}
    for device in ('a100', 'mi210'):
        path = make_run(tmp_path / device, ['code', 'math'], 3)
        meta = read_json(path / 'manifest.json')
        meta['config']['hardware'] = {'vendor': inv.DEVICES[device], 'device': device}
        records, scores = [], []
        for workload in ('code', 'math'):
            data = read_jsonl(path / f'{workload}.jsonl')
            (path / f'{workload}.jsonl').unlink()
            for i, row in enumerate(data):
                row['effective_sampling'] = {'temperature': 0, 'seed': 42}
                if device == 'mi210' and i < 2:
                    row['output_token_ids'] = [4]
                    row['output_text'] = f'candidate {i}'
                records.append(row)
                append_jsonl(path / f'{workload}.jsonl', row)
                correct = (i != 1) if device == 'a100' else (i != 0)
                scores.append({'workload': workload, 'prompt_id': row['prompt_id'], 'status': 'scored',
                               'correct': correct, 'score': float(correct), 'evaluator_version': 'fixture',
                               'source_sha256': row['source_sha256'], 'output_sha256': digest(row['output_text'])})
        identity = serving_identity(meta['config'])
        server = meta['server_provenance']
        server.update(identity=identity, serving_fingerprint=digest(identity))
        meta['run_fingerprint'] = digest({'config': meta['config'], 'sources': meta['sources'],
                                          'server': server['serving_fingerprint'],
                                          'requests': [[r['workload'], r['prompt_id'], r['request_sha256'], r['source_sha256']] for r in records]})
        write_json(path / 'manifest.json', meta)
        write_json(path / 'evaluation.json', {'summary': {'run_fingerprint': meta['run_fingerprint']}, 'records': scores})
        runs[device] = str(path)
    inventory = tmp_path / 'inventory.json'
    write_json(inventory, {'comparisons': [{'id': 'small_a100_vs_mi210', 'candidate_hardware': 'MI210',
                                           'baseline': runs['a100'], 'candidate': runs['mi210']}]})
    target = tmp_path / 'bundle'
    inv.prepare(inventory, target, ['small'], per_group=2)
    return target


@pytest.fixture
def fake_server(monkeypatch):
    starts = []

    @contextmanager
    def owned(config_path, python):
        config = read_json(config_path)
        identity = serving_identity(config)
        starts.append(config)
        write_json(config['server']['metadata_path'], {'identity': identity, 'status': 'ready',
                    'serving_fingerprint': digest(identity), 'instance_id': str(len(starts)),
                    'environment': {'accelerators': [{'name': config['hardware']['device']}], 'packages': {}},
                    'effective_context_limit': config['engine']['max_model_len']})
        yield

    def request(self, path, payload=None):
        if path == '/v1/models':
            return {'data': [{'id': self.config['model']}]}
        token = 3 if self.config['hardware']['vendor'] == 'nvidia' else 4
        choice = {'index': 0, 'text': str(token), 'token_ids': [token], 'finish_reason': 'stop'}
        if 'logprobs' in payload:
            assert payload['return_tokens_as_token_ids'] is True and payload['max_tokens'] == 1
            choice['logprobs'] = {'top_logprobs': [{f'token_id:{token}': -.1, f'token_id:{7-token}': -.2}]}
        return {'choices': [choice],
                'usage': {'completion_tokens': 1, 'prompt_tokens': len(payload['prompt'])},
                'prompt_token_ids': payload['prompt']}

    monkeypatch.setattr(inv, 'owned_server', owned)
    monkeypatch.setattr('driftbench_runner.http_backend.HTTPBackend.request', request)
    return starts


def test_bundle_integrity_and_portability(bundle, tmp_path):
    root, meta = inv.load_bundle(bundle)
    cases = read_json(root / 'models/small/cases.json')
    assert len(cases) == 6
    assert {r['workload'] for r in cases} == {'code', 'math'}
    assert {r['category'] for r in cases} == {'control', 'improvement', 'regression'}
    # Original input runs are no longer needed after prepare.
    import shutil
    shutil.rmtree(tmp_path / 'a100')
    shutil.rmtree(tmp_path / 'mi210')
    moved = tmp_path / 'moved'
    bundle.rename(moved)
    assert inv.load_bundle(moved)[1]['fingerprint'] == meta['fingerprint']
    with (moved / 'triage.csv').open('a') as handle:
        handle.write('changed')
    with pytest.raises(ValueError, match='Bundle content changed'):
        inv.load_bundle(moved)


def test_run_report_resume_and_transferred_trials(bundle, fake_server, tmp_path, monkeypatch):
    output = tmp_path / 'trials'
    for device in ('a100', 'mi210'):
        inv.run_trials(bundle, device, output, __import__('sys').executable, conditions=['original', 'serial', 'serial-eager'], repeats=2)
    assert len(fake_server) == 12
    # Each trial has a valid normal DriftBench dataset/manifest even though the
    # selection is not the original benchmark prefix.
    trials = inv.discover_trials([output])
    assert len(trials) == 12 and all(len(t[3]) == 6 for t in trials)
    monkeypatch.setattr('driftbench_runner.stages.execute_code', lambda row, output: {
        'status': 'scored', 'correct': True, 'score': 1., 'detail': 'fixture',
        'candidate_sha256': digest(inv_code_units(row, output))})
    # Normal stage evaluation can score these settings without a special evaluator.
    inv.evaluate([output])
    for device in ('a100', 'mi210'):
        inv.run_trials(bundle, device, output, __import__('sys').executable, conditions=['original', 'serial', 'serial-eager'], repeats=2, resume=True)
    assert len(fake_server) == 12
    moved = tmp_path / 'copied-results'
    output.rename(moved)
    result = inv.report(bundle, [moved], tmp_path / 'report')
    assert result['evaluation_complete']
    report = read_json(tmp_path / 'report/report.json')
    cross = [r for r in report['agreements'] if r['kind'] == 'cross_device']
    within = [r for r in report['agreements'] if r['kind'] == 'within_device']
    conditions = [r for r in report['agreements'] if r['kind'] == 'condition_change']
    assert cross and all(r['token_agreement_percent'] == 0 for r in cross)
    assert within and all(r['token_agreement_percent'] == 100 for r in within)
    assert any(r['left_condition'] == 'serial' and r['right_condition'] == 'serial-eager' for r in conditions)
    requests = read_json(tmp_path / 'report/probe-cases.json')
    assert len(requests['cases']) == 18
    assert all(r['position'] == 0 and r['input_ids'] == [1, 2] for r in requests['cases'])


def inv_code_units(row, output):
    from driftbench_runner.evaluation import code_units
    return code_units(row['prompt'], output, row['entry_point'])


def test_partial_and_changed_outputs_refused(bundle, fake_server, tmp_path):
    output = tmp_path / 'runs'
    inv.run_trials(bundle, 'a100', output, __import__('sys').executable, conditions=['serial'], repeats=1)
    path, info, _, outputs = inv.discover_trials([output])[0]
    info['status'] = 'failed'
    write_json(path / 'experiment.json', info)
    with pytest.raises(ValueError, match='Incomplete trial'):
        inv.run_trials(bundle, 'a100', output, __import__('sys').executable, conditions=['serial'], repeats=1, resume=True)
    info['status'] = 'complete'
    write_json(path / 'experiment.json', info)
    rows = read_jsonl(path / 'math.jsonl')
    rows[0]['output_token_ids'] = [99]
    (path / 'math.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='Changed response'):
        inv.discover_trials([output])


def test_missing_evaluation_stays_pending(bundle, fake_server, tmp_path):
    inv.run_trials(bundle, 'a100', tmp_path / 'runs', __import__('sys').executable, conditions=['original'], repeats=1)
    result = inv.report(bundle, [tmp_path / 'runs'], tmp_path / 'report')
    assert not result['evaluation_complete']
    assert all(r['accuracy_percent'] is None for r in read_json(tmp_path / 'report/report.json')['accuracy'])


def test_condition_controls_preserve_generation(bundle, tmp_path):
    base = read_json(bundle / 'models/small/a100.json')
    for condition in inv.CONDITIONS:
        config = inv.trial_config(base, tmp_path / condition, 'a100', condition, 8001)
        assert config['generation'] == base['generation'] and config['revision'] == base['revision']
        assert config['engine']['dtype'] == ('float32' if condition.endswith('fp32') else base['engine']['dtype'])
        if condition != 'original':
            assert config['batch_size'] == config['engine']['max_num_seqs'] == 1
    assert base == read_json(bundle / 'models/small/a100.json')


@pytest.mark.parametrize('a,b,expected', [([1], [1], None), ([1], [1, 2], 1), ([2], [], 0), ([1, 3], [1, 4], 1)])
def test_first_difference(a, b, expected):
    assert inv.first_difference(a, b) == expected


def test_topk_parsing_requires_token_ids():
    record = {'server_response': {'choices': [{'logprobs': {'top_logprobs': [{'token_id:10': -.2, 'token_id:2': -.1, 'token_id:3': -.9}]}}]}}
    assert [r['token_id'] for r in probes.parse_topk(record, 2)] == [2, 10]
    bad = deepcopy(record)
    bad['server_response']['choices'][0]['logprobs']['top_logprobs'][0] = {'word': -.2}
    with pytest.raises(ValueError, match='token ID'):
        probes.parse_topk(bad, 2)


def test_tensor_bits_signed_zero_and_shape():
    metric = probes.tensor_metrics(np.array([0., 1.], dtype='float32'), np.array([-0., 1.], dtype='float32'))
    assert not metric['bitwise_equal'] and metric['max_absolute_error'] == 0
    assert metric['bitwise_equal_elements_percent'] == 50
    with pytest.raises(ValueError, match='shape/dtype'):
        probes.tensor_metrics(np.zeros(2, dtype='float32'), np.zeros(2, dtype='float64'))


def test_probe_prefix_mismatch_refused(tmp_path):
    case = {'workload': 'math', 'prompt_id': 'x', 'position': 0, 'input_ids': [1], 'prefix_sha256': digest([1]),
            'top_k': [{'token_id': 2, 'logprob': -.1}, {'token_id': 3, 'logprob': -.2}], 'top1_top2_logprob_margin': .1}
    record = {'status': 'complete', 'bundle_fingerprint': 'b', 'requests_fingerprint': 'r', 'model_key': 'm',
              'condition': 'serial', 'top_k': 2, 'score_type': 'API logprobs', 'device': 'a100', 'records': [case]}
    write_json(tmp_path / 'a.json', record)
    record['records'][0]['input_ids'] = [2]
    write_json(tmp_path / 'b.json', record)
    with pytest.raises(ValueError, match='histories differ'):
        probes.compare_probes(tmp_path / 'a.json', tmp_path / 'b.json', tmp_path / 'out.csv')


def probe_requests(bundle, path):
    _, meta = inv.load_bundle(bundle)
    c = read_json(bundle / 'models/small/cases.json')[0]
    ids = c['request']['input_ids'] + [6]
    requests = {'schema_version': 1, 'bundle_fingerprint': meta['fingerprint'], 'cases': [
        {'model_key': 'small', 'condition': 'serial', 'workload': c['workload'], 'prompt_id': c['prompt_id'],
         'position': 1, 'input_ids': ids, 'prefix_sha256': digest(ids), 'left_token': 3, 'right_token': 4}]}
    requests['fingerprint'] = digest(requests)
    write_json(path, requests)


def test_full_probe_roundtrip(bundle, fake_server, tmp_path):
    requests = tmp_path / 'requests.json'
    probe_requests(bundle, requests)
    for device in ('a100', 'mi210'):
        probes.probe(bundle, requests, device, 'small', 'serial', tmp_path / f'probe-{device}', __import__('sys').executable, top_k=2)
    out = tmp_path / 'topk.csv'
    result = probes.compare_probes(tmp_path / 'probe-a100/probe.json', tmp_path / 'probe-mi210/probe.json', out)
    assert result['paired_probes'] == 1
    import csv
    row = next(csv.DictReader(out.open()))
    assert row['top1_equal'] == 'False' and row['top_k_overlap_percent'] == '100.0'


def test_reference_capture_and_provenance(bundle, tmp_path, monkeypatch):
    import torch
    import transformers
    from types import SimpleNamespace

    class Model:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, input_ids, **kwargs):
            n = input_ids.shape[1]
            hidden = torch.ones((1, n, 4), dtype=torch.bfloat16)
            return SimpleNamespace(hidden_states=(hidden, hidden * 2), logits=torch.arange(4).float().repeat(1, n, 1))

    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: Model())
    requests = tmp_path / 'requests.json'
    probe_requests(bundle, requests)
    for side in ('left', 'right'):
        probes.capture_reference(bundle, requests, 'small', 'serial', tmp_path / side, device='cpu', limit=1)
    result = probes.compare_captures(tmp_path / 'left', tmp_path / 'right', tmp_path / 'capture-diff.json')
    assert all(r['bitwise_equal'] for r in result['arrays'].values())
    record = read_json(tmp_path / 'right/capture.json')
    record['revision'] = 'wrong'
    write_json(tmp_path / 'right/capture.json', record)
    with pytest.raises(ValueError, match='revision'):
        probes.compare_captures(tmp_path / 'left', tmp_path / 'right', tmp_path / 'bad.json')


def test_microbench_reference_and_identity(tmp_path):
    result = probes.microbench('cpu', tmp_path / 'micro')
    assert len(result['operations']) == 16
    with np.load(tmp_path / 'micro/tensors.npz', allow_pickle=False) as arrays:
        assert arrays['reduction_fp32_0'] == 64
        assert arrays['reduction_late_cast_0'] == 64
        assert arrays['reduction_low_0'] == 1
        # The tie above 0.5 rounds to even, immediately above it rounds up.
        assert arrays['bf16_conversion'][:3].tolist() == [.5, .5, .50390625]
    equal = probes.compare_microbench(tmp_path / 'micro', tmp_path / 'micro', tmp_path / 'equal.json')
    assert all(r['bitwise_equal'] for r in equal['arrays'].values())
    import shutil
    shutil.copytree(tmp_path / 'micro', tmp_path / 'other')
    changed = read_json(tmp_path / 'other/microbench.json')
    changed['operations'][0]['inputs_sha256'] = 'different-inputs'
    write_json(tmp_path / 'other/microbench.json', changed)
    with pytest.raises(ValueError, match='inputs or representations differ'):
        probes.compare_microbench(tmp_path / 'micro', tmp_path / 'other', tmp_path / 'bad.json')
