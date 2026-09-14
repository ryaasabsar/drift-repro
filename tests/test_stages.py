import json
from pathlib import Path
import subprocess

import pytest

from driftbench_runner import evaluation, safety, stages, transfer
from driftbench_runner.common import ROOT, WORKLOADS, append_jsonl, digest, load_workloads, read_json, write_json


def make_run(path, workloads=None, limit=2):
    path.mkdir(parents=True)
    rows, sources = load_workloads(workloads or list(WORKLOADS), limit)
    config = read_json(ROOT / 'configs/rtx3060-qwen35-vllm.json')
    config['setup_id'] = 'fixture_a100'
    config['hardware'] = {'vendor': 'nvidia', 'device': 'A100 fixture'}
    records = []
    for row in rows:
        record = {**{k: row[k] for k in ('workload', 'prompt_id', 'prompt', 'source_sha256')},
                  'rendered_prompt': row['prompt'], 'input_token_ids': [1, 2],
                  'request_sha256': digest({'input_ids': [1, 2], 'prompt': row['prompt']}),
                  'output_text': row.get('canonical_solution', 'fixture output'),
                  'output_token_ids': [3], 'input_tokens': 2, 'output_tokens': 1, 'status': 'ok',
                  'finish_reason': 'stop', 'latency_seconds': 1.0}
        records.append(record)
        append_jsonl(path / f'{row["workload"]}.jsonl', record)
    identity = {'config': config, 'sources': sources,
                'requests': [[r['workload'], r['prompt_id'], r['request_sha256'], r['source_sha256']] for r in records]}
    write_json(path / 'manifest.json', {'status': 'complete', 'config': config, 'sources': sources,
               'run_fingerprint': digest(identity), 'expected_records': len(records),
               'environment': {'gpu_name': 'A100 fixture', 'packages': {}}})
    return path


@pytest.fixture
def fake_workers(monkeypatch):
    calls = []
    def code(row, output):
        calls.append(('code', row['entry_point']))
        return {'status': 'scored', 'correct': True, 'score': 1.0, 'detail': 'fixture'}
    def judge(directory, output, model, revision, device):
        calls.append(('safety', str(directory)))
        manifest, outputs = evaluation.load_run(directory)
        existing = {r['prompt_id'] for r in stages.safety_records(directory, outputs).values()}
        for key, row in outputs.items():
            if key[0] == 'safety' and key[1] not in existing:
                append_jsonl(output, {'workload': 'safety', 'prompt_id': key[1], 'source_sha256': row['source_sha256'],
                    'output_sha256': digest(row['output_text']), 'label': 'safe', 'severity': 'safe',
                    'judge': {'model': model, 'revision': revision, 'environment': {'device_mode': device, 'host': 'fixed-judge'}}})
    monkeypatch.setattr(stages, 'execute_code', code)
    monkeypatch.setattr(safety, 'judge_safety', judge)
    monkeypatch.setattr('driftbench_runner.preset.load_credentials', lambda: None)
    return calls


@pytest.mark.parametrize('first,second', [('code', 'safety'), ('safety', 'code')])
def test_round_trip_both_stage_orders_without_vendor_checkout(tmp_path, monkeypatch, fake_workers, first, second):
    source = make_run(tmp_path / 'source')
    original = (source / 'manifest.json').read_bytes()
    raw = {p.name: p.read_bytes() for p in source.glob('*.jsonl')}
    stages.run_stage(source, first)
    archive = tmp_path / 'after-first.tar.gz'
    transfer.pack(source, archive)
    monkeypatch.setattr(evaluation, 'DATASET_DIR', tmp_path / 'no-vendor-on-receiver')
    monkeypatch.setattr(transfer, 'DATASET_DIR', tmp_path / 'no-vendor-on-receiver')
    imported = tmp_path / 'machine-two' / 'renamed-run'
    transfer.unpack(archive, imported)
    stages.run_stage(imported, second)
    second_archive = tmp_path / 'after-both.tar.gz'
    transfer.pack(imported, second_archive)
    final = tmp_path / 'machine-three' / 'final'
    transfer.unpack(second_archive, final)
    count = len(fake_workers)
    result = stages.run_stage(final, 'final')
    assert result['status'] == 'complete'
    assert len(fake_workers) == count  # Final never invokes code or a safety model.
    assert (final / 'manifest.json').read_bytes() == original
    assert all((final / name).read_bytes() == content for name, content in raw.items())
    scored = read_json(final / 'evaluation.json')
    assert scored['summary']['workloads']['code']['scored'] == 2
    assert scored['summary']['workloads']['safety']['scored'] == 2
    assert scored['summary']['workloads']['chat']['pairwise_only'] == 2
    assert 'A100 fixture' in (final / 'tables/rows.csv').read_text()
    stages.run_stage(final, 'code')
    assert len(fake_workers) == count  # Completed code stage can be reused without Bubblewrap.


def test_final_refuses_missing_stages_and_status_is_derived(tmp_path, fake_workers):
    run = make_run(tmp_path / 'run')
    write_json(run / 'stages.json', {'status': 'complete'})
    with pytest.raises(ValueError, match='Complete the code and safety'):
        stages.run_stage(run, 'final')
    assert stages.stage_status(run)['status'] == 'needs_evaluation'
    stages.run_stage(run, 'code')
    with pytest.raises(ValueError, match='Complete the code and safety'):
        stages.run_stage(run, 'final')


def test_partial_code_resume_and_real_reason_preserved(tmp_path, monkeypatch, fake_workers):
    run = make_run(tmp_path / 'run', ['code'])
    calls = []
    def interrupted(row, output):
        calls.append(row['entry_point'])
        if len(calls) == 3:
            raise KeyboardInterrupt
        return {'status': 'scored', 'correct': True, 'score': 1.0}
    monkeypatch.setattr(stages, 'execute_code', interrupted)
    with pytest.raises(KeyboardInterrupt):
        stages.run_stage(run, 'code')
    assert stages.stage_status(run)['settings']['.']['code']['completed'] == 1
    monkeypatch.setattr(stages, 'execute_code', lambda *args: {'status': 'pending', 'reason': 'bwrap: Operation not permitted'})
    with pytest.raises(RuntimeError, match='bwrap: Operation not permitted'):
        stages.run_stage(run, 'code')
    calls.clear()
    monkeypatch.setattr(stages, 'execute_code', interrupted)
    stages.run_stage(run, 'code')
    assert len(calls) == 2  # Fixture preflight plus only the missing response.


def test_stale_labels_and_code_scores_rejected(tmp_path, fake_workers):
    run = make_run(tmp_path / 'run')
    stages.run_stage(run, 'code')
    file = run / stages.CODE_FILE
    original = file.read_bytes()
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    rows[0]['output_sha256'] = 'changed'
    file.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with pytest.raises(ValueError, match='stale saved code'):
        stages.stage_status(run)
    file.write_bytes(original)
    stages.run_stage(run, 'safety')
    file = run / stages.SAFETY_FILE
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    rows[0]['output_sha256'] = 'changed'
    file.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with pytest.raises(ValueError, match='stale safety'):
        stages.stage_status(run)


def test_suite_import_updates_status_without_changing_original_paths(tmp_path, fake_workers):
    suite = tmp_path / 'suite'
    run = make_run(suite / 'settings' / 'fixture_a100', ['code', 'safety'])
    plan = {'settings': [{'id': 'fixture_a100', 'run_dir': '/old-host/results/fixture_a100'}]}
    write_json(suite / 'suite.json', {'status': 'inference_complete', 'plan': plan,
               'settings': {'fixture_a100': {'status': 'inference_complete'}}})
    stages.run_stage(suite, 'code')
    stages.run_stage(suite, 'safety')
    stages.run_stage(suite, 'final')
    saved = read_json(suite / 'suite.json')
    assert saved['status'] == 'complete'
    assert saved['settings']['fixture_a100']['status'] == 'complete'
    assert saved['plan'] == plan
    assert (suite / 'collection/rows.csv').is_file()


def test_incomplete_and_tampered_inference_cannot_be_packed(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    path = run / 'manifest.json'
    manifest = read_json(path)
    manifest['status'] = 'running'
    write_json(path, manifest)
    with pytest.raises(ValueError, match='Finish inference'):
        transfer.pack(run, tmp_path / 'run.tar.gz')
    manifest['status'] = 'complete'
    write_json(path, manifest)
    file = run / 'math.jsonl'
    records = [json.loads(line) for line in file.read_text().splitlines()]
    records[0]['input_token_ids'] = [999]
    file.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    with pytest.raises(ValueError, match='request trace'):
        transfer.pack(run, tmp_path / 'run.tar.gz')


def test_script_dry_run_has_no_credentials_gpu_or_output(tmp_path):
    output = tmp_path / 'runs'
    result = subprocess.run(['bash', str(ROOT / 'scripts/results.sh'), 'infer', '--config',
                             str(ROOT / 'suites/a100-qwen25-7b-llamaguard3.json'), '--run-id', 'a100-r01',
                             '--output-root', str(output), '--limit', '1', '--dry-run'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['expected_records_per_setting'] == 5
    assert not output.exists()


def test_real_humaneval_stage_and_portable_final(tmp_path):
    probe = evaluation.execute_code({'prompt': 'def f():\n', 'entry_point': 'f',
                                    'test_cases': 'def check(f):\n    assert f() == 1'}, '    return 1\n')
    if probe['status'] == 'pending':
        pytest.skip(probe['reason'])
    run = make_run(tmp_path / 'run', ['code'], limit=1)
    stages.run_stage(run, 'code')
    archive = tmp_path / 'code.tar.gz'
    transfer.pack(run, archive)
    imported = tmp_path / 'new-host'
    transfer.unpack(archive, imported)
    stages.run_stage(imported, 'final')
    assert read_json(imported / 'evaluation.json')['summary']['workloads']['code']['correct'] == 1


def test_infer_dispatch_defers_all_evaluation_and_records_stages(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from driftbench_runner import cli, suite
    loaded = []
    monkeypatch.setattr('driftbench_runner.preset.load_credentials', lambda: loaded.append(True))
    def run(config_path, output, **options):
        assert options['inference_only'] is True
        assert options['resume'] is False
        assert options['limit'] == 1
        path = make_run(output / 'settings' / 'fixture_a100', ['math'], 1)
        write_json(output / 'suite.json', {'status': 'inference_complete',
                   'plan': {'settings': [{'id': 'fixture_a100'}]},
                   'settings': {'fixture_a100': {'status': 'inference_complete'}}})
        return {'status': 'inference_complete'}
    monkeypatch.setattr(suite, 'run_suite', run)
    args = SimpleNamespace(command='infer', config=str(ROOT / 'suites/a100-qwen25-7b-llamaguard3.json'),
                           run_id='a100-r01', output_root=str(tmp_path), limit=1, workloads=None, device=None,
                           dry_run=False, resume=False, allow_experimental=False)
    cli.dispatch(args)
    assert loaded == [True]
    assert read_json(tmp_path / 'a100-r01/stages.json')['status'] == 'needs_evaluation'
    write_json(tmp_path / 'a100-r01/transfer.json', {'imported': True})
    args.resume = True
    with pytest.raises(ValueError, match='Imported runs'):
        cli.dispatch(args)
