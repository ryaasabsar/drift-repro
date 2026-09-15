"""Discover saved settings and compare any chosen baseline against explicit candidates."""
import csv
from pathlib import Path

from .common import digest, read_json, write_json
from .comparison import compare_runs
from .stages import setting_directories, verify_inference


def catalog(roots):
    entries = []
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Result directory does not exist: {root}')
        manifests = [root / 'manifest.json'] if (root / 'manifest.json').is_file() else sorted(root.rglob('manifest.json'))
        for path in manifests:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            manifest = read_json(path)
            config = manifest.get('config', {})
            if 'setup_id' not in config:
                continue
            entries.append({'path': str(path.parent), 'setting': config['setup_id'],
                            'model': config['model'], 'framework': config['backend'],
                            'hardware': config.get('hardware'), 'status': manifest['status'],
                            'evaluation_available': (path.parent / 'evaluation.json').is_file(),
                            'run_fingerprint': manifest['run_fingerprint']})
    return entries


def comparison_plan(baseline, candidates):
    baseline = Path(baseline).resolve()
    if not (baseline / 'manifest.json').is_file():
        raise ValueError('Choose one baseline setting directory (use catalog to find paths)')
    paths = []
    for candidate in candidates:
        for path in setting_directories(candidate):
            if path != baseline and path not in paths:
                paths.append(path)
    if not paths:
        raise ValueError('Choose at least one candidate other than the baseline')
    for path in [baseline, *paths]:
        verify_inference(path)
    return {'baseline': str(baseline), 'candidates': [str(p) for p in paths]}


def compare_many(baseline, candidates, output, semantic=False, allow_confounded=False, dry_run=False):
    plan = comparison_plan(baseline, candidates)
    if dry_run:
        return plan
    output = Path(output)
    if output.exists():
        raise ValueError('Comparison output already exists; choose a new directory')
    output.mkdir(parents=True)
    state = {**plan, 'status': 'running', 'comparisons': []}
    summary = []
    try:
        for candidate in plan['candidates']:
            directory = output / (Path(candidate).name + '-' + digest(candidate)[:12])
            try:
                report = compare_runs(plan['baseline'], candidate, directory, semantic, allow_confounded)
            except ValueError as exc:
                # Keep explicit incompatibilities visible, never count them as zero drift.
                state['comparisons'].append({'candidate': candidate, 'status': 'incompatible', 'error': str(exc)})
                continue
            state['comparisons'].append({'candidate': candidate, 'status': 'complete', 'report': str(directory / 'comparison.json')})
            for workload, metrics in report['workloads'].items():
                summary.append({'baseline': plan['baseline'], 'candidate': candidate, 'workload': workload, **metrics})
        state['status'] = 'complete' if all(p['status'] == 'complete' for p in state['comparisons']) else 'incompatible'
    except BaseException:
        state['status'] = 'failed'
        raise
    finally:
        write_json(output / 'comparisons.json', state)
        if summary:
            with (output / 'comparisons.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    return state
