"""Inspect the actual serving interpreters, independent of the tokenizer client."""
import json
import os
from pathlib import Path
import subprocess

from .common import ROOT, read_json
from .hardware import discover, require_hardware
from .software import check_runtime


def doctor(config=None, suite=None, frameworks=None):
    if config and suite:
        raise ValueError('Use --config or --suite, not both')
    if frameworks and not suite:
        raise ValueError('--framework requires --suite')
    if suite:
        from .suite import load_plan
        plan = load_plan(suite, ROOT / 'results/doctor', frameworks=frameworks)
        original = read_json(suite)
        configs = {item['id']: str(Path(suite).resolve().parent / item['config']) for item in original['settings']}
        reports = []
        seen = set()
        for item in plan['settings']:
            key = (item['server_python'], json.dumps(item['config'].get('launch', {}).get('env', {}), sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            try:
                process = subprocess.run([item['server_python'], '-m', 'driftbench_runner', 'doctor', '--config', configs[item['id']]],
                    cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)}, text=True, capture_output=True, timeout=150)
                report = json.loads(process.stdout)
                if process.returncode and report.get('status') != 'failed':
                    report.update(status='failed', error=f'Doctor exited {process.returncode}')
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                report = {'status': 'failed', 'error': str(exc)}
            reports.append({'server_python': item['server_python'], 'representative_setting': item['id'], **report})
        return {'status': 'passed' if all(r['status'] == 'passed' for r in reports) else 'failed', 'environments': reports}
    if config:
        from .serving import configure_serving_environment
        spec = read_json(config)
        try:
            configure_serving_environment(spec)
        except (OSError, ValueError, RuntimeError) as exc:
            return {'status': 'failed', 'error': str(exc)}
    environment = discover()
    if not config:
        return environment
    audit = check_runtime(spec, environment)
    try:
        require_hardware(spec['hardware']['vendor'], environment)
        error = None
    except RuntimeError as exc:
        error = str(exc)
    return {'status': 'passed' if error is None and audit['status'] == 'match' else 'failed',
            'environment': environment, 'runtime_contract': audit, 'error': error}
