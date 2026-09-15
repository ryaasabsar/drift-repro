"""Portable, independently resumable evaluation stages for complete inference runs."""
from contextlib import contextmanager, ExitStack
import fcntl
from pathlib import Path
import re

from .common import ROOT, append_jsonl, digest, keyed, now, read_json, read_jsonl, write_json
from .evaluation import (EVALUATOR_VERSION, code_environment, evaluate_run, execute_code,
                         load_run, run_workloads, saved_code_results)
from .logging import Progress, event
from .scoring import evaluation_complete
from .tables import collect_runs, export_run
from .comparison import verified_evaluations

CODE_FILE = 'code-results.jsonl'
SAFETY_FILE = 'safety-labels.jsonl'


def setting_directories(directory):
    directory = Path(directory).resolve()
    if (directory / 'manifest.json').is_file():
        return [directory]
    if (directory / 'suite.json').is_file():
        spec = read_json(directory / 'suite.json')
        ids = [item['id'] for item in spec['plan']['settings']]
        if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', sid) for sid in ids):
            raise ValueError('Invalid suite setting IDs')
        paths = [directory / 'settings' / sid for sid in ids]
        if not paths or any(not (p / 'manifest.json').is_file() for p in paths):
            raise ValueError('Finish inference for every suite setting, or select one complete setting directory')
        return paths
    raise ValueError(f'Expected a setting or suite result directory: {directory}')


@contextmanager
def result_lock(directory):
    """Use inference's locks as well as a suite-wide lock, without following copied paths."""
    root = Path(directory).resolve()
    paths = setting_directories(root)
    with ExitStack() as stack:
        locks = ([root / '.suite.lock'] if (root / 'suite.json').exists() else [])
        locks += [p / '.run.lock' for p in paths]
        for path in locks:
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another inference/evaluation/transfer operation owns this run') from None
        yield paths


def verify_inference(directory):
    manifest, outputs = load_run(directory)
    sources, info = run_workloads(directory, list(manifest['sources']))
    expected = {}
    for workload, source in manifest['sources'].items():
        if info[workload]['sha256'] != source['sha256']:
            raise ValueError(f'Benchmark file changed: {workload}')
        count = source['selected']
        if type(count) is not int or not 0 < count <= info[workload]['available']:
            raise ValueError('Invalid selected workload count')
        expected.update(keyed([r for r in sources if r['workload'] == workload][:count]))
    if manifest['status'] != 'complete' or manifest['expected_records'] != len(expected) or set(outputs) != set(expected):
        raise ValueError('Finish inference before evaluating or packing this run')
    requests = []
    for key, source in expected.items():
        row = outputs[key]
        if row['status'] != 'ok' or row['source_sha256'] != source['source_sha256'] or row['prompt'] != source['prompt']:
            raise ValueError(f'Invalid inference response source: {key}')
        if row['request_sha256'] != digest({'input_ids': row['input_token_ids'], 'prompt': row['rendered_prompt']}):
            raise ValueError(f'Invalid request trace: {key}')
        requests.append([*key, row['request_sha256'], row['source_sha256']])
    identity = {'config': manifest['config'], 'sources': manifest['sources'], 'requests': requests}
    if manifest['config'].get('transport', 'offline') == 'http':
        server = manifest['server_provenance']
        if digest(server['identity']) != server['serving_fingerprint']:
            raise ValueError('Invalid serving provenance')
        identity['server'] = server['serving_fingerprint']
    if digest(identity) != manifest['run_fingerprint']:
        raise ValueError('Invalid inference fingerprint')
    return manifest, outputs, expected


def safety_records(directory, outputs):
    records = keyed(read_jsonl(Path(directory) / SAFETY_FILE))
    judge = None
    for key, row in records.items():
        source = outputs.get(key)
        current = row.get('judge', {})
        if (key[0] != 'safety' or source is None or row.get('source_sha256') != source['source_sha256'] or
                row.get('output_sha256') != digest(source['output_text']) or row.get('label') not in ('safe', 'unsafe') or
                not current.get('environment') or not re.fullmatch(r'[0-9a-f]{40}', current.get('revision', ''))):
            raise ValueError(f'Invalid or stale safety label: {key}')
        if judge is not None and judge != current:
            raise ValueError('Safety labels mix different judge environments')
        judge = current
    return records


def stage_status(directory):
    reports = {}
    root = Path(directory).resolve()
    for path in setting_directories(root):
        manifest, outputs, _ = verify_inference(path)
        code = saved_code_results(path / CODE_FILE, manifest, outputs)
        safety = safety_records(path, outputs)
        scores = verified_evaluations(path, manifest, outputs)
        final = evaluation_complete(path)
        for key in outputs:
            score = scores.get(key, {})
            if key[0] == 'code':
                final = final and key in code and all(score.get(k) == v for k, v in code.get(key, {}).items())
            elif key[0] == 'safety':
                final = final and key in safety and all(score.get(k) == safety.get(key, {}).get(k)
                                                       for k in ('judge', 'label', 'severity'))
        def status(workload, records):
            count = sum(key[0] == workload for key in outputs)
            return {'status': 'not_required' if not count else 'complete' if len(records) == count else 'pending',
                    'completed': len(records), 'expected': count}
        reports[str(path.relative_to(root))] = {
            'run_fingerprint': manifest['run_fingerprint'], 'inference': 'complete',
            'code': status('code', code), 'safety': status('safety', safety),
            'final': 'complete' if final else 'pending'}
    return {'schema_version': 1, 'settings': reports,
            'status': 'complete' if all(r['final'] == 'complete' for r in reports.values()) else 'needs_evaluation'}


def refresh_status(directory):
    root = Path(directory).resolve()
    report = stage_status(root)
    write_json(root / 'stages.json', {**report, 'updated_at': now()})
    if (root / 'suite.json').exists():
        state = read_json(root / 'suite.json')
        for relative, stages in report['settings'].items():
            sid = Path(relative).name
            state['settings'][sid].update(status='complete' if stages['final'] == 'complete' else 'inference_complete',
                                          evaluation_stages=stages)
        state.update(status=report['status'], updated_at=now())
        # Inference plan/provenance stays unchanged, including original host paths.
        write_json(root / 'suite.json', state)
    return report


def code_stage(directory):
    manifest, outputs, sources = verify_inference(directory)
    path = Path(directory) / CODE_FILE
    existing = saved_code_results(path, manifest, outputs)
    selected = {key: row for key, row in outputs.items() if key[0] == 'code'}
    if not selected:
        return
    if len(existing) == len(selected):
        event('Reusing completed code evaluation', stage='code', total=len(existing))
        return
    environment = code_environment()
    if any(row['execution_environment'] != environment for row in existing.values()):
        raise ValueError('Resume code evaluation on the same evaluator environment')
    check = execute_code({'prompt': 'def check_value():\n', 'entry_point': 'check_value',
                          'test_cases': 'def check(candidate):\n    assert candidate() == 1'}, '    return 1\n')
    if not check.get('correct'):
        raise RuntimeError('HumanEval sandbox unavailable: ' + check.get('reason', check.get('detail', str(check))))
    with Progress('Evaluating saved code', stage='code', total=len(selected), completed=len(existing)) as progress:
        for key, row in selected.items():
            if key in existing:
                continue
            result = execute_code(sources[key], row['output_text'])
            if result['status'] != 'scored':
                raise RuntimeError(result.get('reason', 'Code evaluation unavailable'))
            record = {'workload': key[0], 'prompt_id': key[1], 'run_fingerprint': manifest['run_fingerprint'],
                      'source_sha256': row['source_sha256'], 'output_sha256': digest(row['output_text']),
                      'evaluator_version': EVALUATOR_VERSION, 'method': 'humaneval_execution_pass_at_1',
                      'execution_environment': environment, **result}
            append_jsonl(path, record)
            existing[key] = record
            progress.update(len(existing), prompt_id=key[1])


def run_stage(directory, stage, judge_model=None, judge_revision=None, judge_device='cuda'):
    root = Path(directory).resolve()
    if stage == 'evaluate':
        # RTX stage: execute isolated code, then score all saved workloads.
        # Check safety first so a missing A100 stage fails before code execution.
        report = stage_status(root)
        if any(s['safety']['status'] == 'pending' for s in report['settings'].values()):
            raise ValueError('Complete stage safety on A100 before stage evaluate on RTX')
        run_stage(root, 'code')
        return run_stage(root, 'final')
    if stage == 'status':
        return stage_status(root)
    with result_lock(root) as paths:
        # Validate every setting before starting expensive work.
        stage_status(root)
        if stage == 'safety':
            from .credentials import load_credentials
            from .safety import judge_safety
            lock = read_json(ROOT / 'models.lock.json')
            pinned = lock['safety_llamaguard3_8b']
            model = judge_model or pinned['model']
            revision = judge_revision or next((v['revision'] for v in lock.values() if v['model'] == model), None)
            if not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
                raise ValueError('Specify an immutable --judge-revision for this model')
            load_credentials()
        try:
            for path in paths:
                manifest, outputs, _ = verify_inference(path)
                if stage == 'safety':
                    if 'safety' in manifest['sources']:
                        judge_safety(path, path / SAFETY_FILE, model, revision, judge_device)
                        if len(safety_records(path, outputs)) != sum(k[0] == 'safety' for k in outputs):
                            raise RuntimeError('Safety evaluation remains incomplete')
                elif stage == 'code':
                    code_stage(path)
                elif stage == 'final':
                    stages = stage_status(path)['settings']['.']
                    if any(stages[s]['status'] == 'pending' for s in ('code', 'safety')):
                        raise ValueError('Complete the code and safety stages before final scoring (on their respective hosts)')
                    evaluate_run(path, safety_labels=path / SAFETY_FILE, code_results=path / CODE_FILE)
                    if not evaluation_complete(path):
                        raise RuntimeError('Final evaluation remains incomplete')
                    export_run(path)
                else:
                    raise ValueError(f'Unknown stage: {stage}')
            if stage == 'final' and (root / 'suite.json').exists():
                collect_runs(paths, root / 'collection')
        finally:
            refresh_status(root)
        return stage_status(root)
