"""Read-only snapshots of saved progress; timestamps do not assert process liveness."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .common import read_json
from .logging import clean, duration, render


def saved_lines(path):
    if not path.exists():
        return 0
    with path.open('rb') as handle:
        return sum(line.endswith(b'\n') and bool(line.strip()) for line in handle)


def last_event(directory):
    latest = None
    for path in (directory / 'logs').glob('*.events.jsonl'):
        # Read a bounded tail; a partial write is not an event yet.
        with path.open('rb') as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - 65536))
            if size > 65536:
                handle.readline()
            for line in handle:
                if not line.endswith(b'\n'):
                    continue
                try:
                    record = json.loads(line)
                    if not all(k in record for k in ('timestamp', 'message', 'stage', 'level')):
                        continue
                    if record['level'] == 'debug':
                        continue
                    if latest is None or record['timestamp'] > latest['timestamp']:
                        latest = record
                except (ValueError, UnicodeError):
                    continue
    return latest


def run_snapshot(directory, expected=None):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    sources = manifest.get('sources', expected or {})
    report = {'run': str(directory), 'inference_status': manifest.get('status', 'not_started'),
              'workloads': {name: {'saved': saved_lines(directory / f'{name}.jsonl'), 'expected': info['selected']}
                            for name, info in sources.items()}, 'last_event': last_event(directory)}
    report['safety_judgments'] = saved_lines(directory / 'safety-labels.jsonl')
    evaluation = directory / 'evaluation.json'
    if evaluation.exists():
        report['evaluation'] = read_json(evaluation)['summary']['workloads']
        report['evaluation_matches_saved_counts'] = all(
            report['evaluation'].get(workload, {}).get('generated') == counts['saved']
            for workload, counts in report['workloads'].items())
    return report


def snapshot(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'No result directory: {directory}')
    if not (directory / 'suite.json').exists():
        return {'kind': 'run', **run_snapshot(directory)}
    state = read_json(directory / 'suite.json')
    settings = {}
    for item in state['plan']['settings']:
        sid = item['id']
        # Resolve within the supplied folder so copied suites remain inspectable.
        settings[sid] = {**run_snapshot(directory / 'settings' / sid, state['plan']['sources']),
                         'saved_status': state['settings'][sid]['status'],
                         'error': state['settings'][sid].get('error')}
    return {'kind': 'suite', 'run': str(directory), 'saved_status': state['status'],
            'settings': settings, 'last_event': last_event(directory), 'error': state.get('error'),
            'collection_error': state.get('collection_error')}


def format_snapshot(report):
    lines = [f"{report['kind'].capitalize()}: {report['run']}",
             f"Saved status: {report.get('saved_status', report.get('inference_status'))}"]
    latest = report.get('last_event')
    if latest:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(latest['timestamp'])).total_seconds()
        lines += [f'Last recorded activity ({duration(age)} ago; this does not confirm the process is still running):', render(latest)]
    else:
        lines.append('No runner events recorded yet.')
    settings = report.get('settings', {report['run']: report})
    for sid, value in settings.items():
        counts = value['workloads'].values()
        saved = sum(c['saved'] for c in counts)
        expected = sum(c['expected'] for c in counts)
        state = value.get('saved_status', value.get('inference_status'))
        lines.append(f'  {clean(sid)}: {state} | {saved}/{expected} responses saved')
        lines.append('    ' + ' | '.join(f"{w} {c['saved']}/{c['expected']}" for w, c in value['workloads'].items()))
        if 'safety' in value['workloads']:
            lines.append(f"    Safety judgments saved: {value['safety_judgments']}/{value['workloads']['safety']['saved']}")
        if 'evaluation' in value:
            scored = sum(w.get('scored', 0) for w in value['evaluation'].values())
            pending = sum(w.get('pending', 0) for w in value['evaluation'].values())
            paired = sum(w.get('pairwise_only', 0) for w in value['evaluation'].values())
            lines.append(f'    Evaluation: {scored} scored | {pending} pending | {paired} pairwise only')
        if value.get('error'):
            lines.append('    Error: ' + clean(value['error']))
    for key in ('error', 'collection_error'):
        if report.get(key):
            lines.append(f'{key}: {clean(report[key])}')
    return '\n'.join(lines)


def show_status(directory, as_json=False, watch=False, interval=15):
    previous = None
    while True:
        report = snapshot(directory)
        if report != previous:
            if as_json:
                print(json.dumps(report, indent=None if watch else 2), flush=True)
            else:
                print(format_snapshot(report), flush=True)
            previous = report
        if not watch:
            return report
        time.sleep(interval)
