from copy import deepcopy

import pytest

from driftbench_runner.common import digest, read_json, write_json
from driftbench_runner.comparison import compare_runs
from driftbench_runner.evaluation import load_run
from test_stages import make_run


@pytest.fixture
def judged_runs(tmp_path):
    judge = {
        'model': 'meta-llama/Llama-Guard-3-8B', 'revision': 'fixed-revision',
        'template': 'official_user_and_assistant', 'family': 'llamaguard3',
        'binary_policy': 'native_binary',
        'decoding': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128},
        'reproducibility': {'seed': 42, 'deterministic_algorithms': True},
        'dtype': 'bfloat16', 'attention_implementation': 'eager',
        'environment': {'device_mode': 'cuda', 'gpu_name': 'MI210'},
    }
    other = deepcopy(judge)
    other.update(dtype='backend_defined', attention_implementation='backend_defined',
                 reproducibility={'seed': 42, 'determinism': 'not_established'},
                 environment={'device_mode': 'tenstorrent', 'gpu_name': 'P150b'},
                 serving_fingerprint='tt-server',
                 generation={'temperature': 0.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0,
                             'max_tokens': 128, 'repetition_penalty': 1.0,
                             'presence_penalty': 0.0, 'frequency_penalty': 0.0})
    paths = []
    for name, metadata in [('baseline', judge), ('candidate', other)]:
        path = make_run(tmp_path / name, ['safety'], limit=1)
        manifest, outputs = load_run(path)
        # Hold all inference controls fixed to isolate evaluator validation.
        manifest['client_environment'] = {'packages': {}}
        write_json(path / 'manifest.json', manifest)
        records = [{'workload': key[0], 'prompt_id': key[1], 'status': 'scored',
                    'source_sha256': row['source_sha256'], 'output_sha256': digest(row['output_text']),
                    'method': 'safety_classification', 'evaluator_version': 'fixture-v1',
                    'label': 'safe', 'correct': True, 'score': 1.0, 'judge': metadata}
                   for key, row in outputs.items()]
        write_json(path / 'evaluation.json', {'summary': {'run_fingerprint': manifest['run_fingerprint']},
                                              'records': records})
        paths.append(path)
    return paths


def edit_score(path, change):
    evaluation = read_json(path / 'evaluation.json')
    change(evaluation['records'][0])
    write_json(path / 'evaluation.json', evaluation)


def test_runtime_override_preserves_both_judges(judged_runs, tmp_path):
    with pytest.raises(ValueError, match='Evaluators differ.*judge'):
        compare_runs(*judged_runs, tmp_path / 'strict')
    report = compare_runs(*judged_runs, tmp_path / 'allowed', allow_confounded=True)
    row = report['records'][0]
    assert report['confounds'] == ['safety_judge_runtime']
    assert row['judge_runtime_changed'] and not row['label_flip']
    assert 'safety_judge' not in row
    for side, path in zip(('baseline', 'candidate'), judged_runs):
        assert row[f'{side}_safety_judge'] == read_json(path / 'evaluation.json')['records'][0]['judge']
    assert 'generation' in row['safety_judge_differences']
    assert report['workloads']['safety']['judge_runtime_changed_pairs'] == 1
    assert 'cannot be attributed solely' in (tmp_path / 'allowed/rows.html').read_text()


@pytest.mark.parametrize('field,value', [
    ('model', 'other-model'), ('revision', 'other-revision'), ('template', 'other-template'),
    ('family', 'other-family'), ('binary_policy', 'other-policy'),
    ('decoding.max_new_tokens', 64), ('reproducibility.seed', 7),
    ('generation.temperature', 0.5), ('generation.max_tokens', 64),
    ('generation.frequency_penalty', 0.5), ('new_labeling_option', True),
    ('revision', None), ('reproducibility.seed', None),
])
def test_override_rejects_changed_or_missing_protocol(judged_runs, tmp_path, field, value):
    def change(row):
        target = row['judge']
        parts = field.split('.')
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    edit_score(judged_runs[1], change)
    with pytest.raises(ValueError, match='Evaluators differ.*judge'):
        compare_runs(*judged_runs, tmp_path / 'rejected', allow_confounded=True)


@pytest.mark.parametrize('field,value,error', [
    ('method', 'another-method', 'Evaluators differ'),
    ('evaluator_version', 'another-version', 'Evaluators differ'),
    ('execution_environment', {'host': 'different-code-host'}, 'Evaluators differ'),
    ('output_sha256', 'stale-output', 'Stale evaluation'),
    ('source_sha256', 'stale-source', 'Stale evaluation source'),
])
def test_other_evaluator_and_integrity_checks_stay_strict(judged_runs, tmp_path, field, value, error):
    edit_score(judged_runs[1], lambda row: row.update({field: value}))
    with pytest.raises(ValueError, match=error):
        compare_runs(*judged_runs, tmp_path / 'rejected', allow_confounded=True)


def test_identical_response_disagreement_is_explicit(judged_runs, tmp_path):
    edit_score(judged_runs[1], lambda row: row.update(label='unsafe', correct=False, score=0.0))
    report = compare_runs(*judged_runs, tmp_path / 'allowed', allow_confounded=True)
    row = report['records'][0]
    assert not row['text_changed'] and row['label_flip']
    assert row['identical_response_label_disagreement']
    assert report['workloads']['safety']['identical_response_label_disagreements'] == 1
    assert report['workloads']['safety']['label_flips'] == 1
    # The same judge disagreeing with itself remains a consistency error.
    judge = read_json(judged_runs[0] / 'evaluation.json')['records'][0]['judge']
    edit_score(judged_runs[1], lambda row: row.update(judge=judge))
    with pytest.raises(ValueError, match='Identical response received different labels'):
        compare_runs(*judged_runs, tmp_path / 'rejected', allow_confounded=True)


def test_matching_judge_remains_strictly_comparable(judged_runs, tmp_path):
    judge = read_json(judged_runs[0] / 'evaluation.json')['records'][0]['judge']
    edit_score(judged_runs[1], lambda row: row.update(judge=judge))
    report = compare_runs(*judged_runs, tmp_path / 'same')
    assert report['confounds'] == []
    assert report['records'][0]['safety_judge'] == judge
    assert not report['records'][0]['judge_runtime_changed']
