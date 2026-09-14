import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from driftbench_runner import logging as logs
from driftbench_runner.common import ROOT, write_json
from driftbench_runner.status import snapshot, format_snapshot
from driftbench_runner.suite import run_command


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    for handler in logs.LOGGER.handlers[:]:
        logs.LOGGER.removeHandler(handler)
        handler.close()
    logs._INTERVAL = 15


def test_console_stderr_and_verbose_files_are_separate(tmp_path, capsys):
    readable, structured = tmp_path / 'run.log', tmp_path / 'events.jsonl'
    logs.configure('warning', readable, structured)
    logs.event('Internal detail', stage='startup', level='debug')
    logs.event('Waiting\nfor server\x1b[31m', stage='startup', setting='gpu_model', level='warning')
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'Internal detail' not in captured.err
    assert 'WARNING' in captured.err and '[gpu_model]' in captured.err
    assert len(captured.err.splitlines()) == 1
    assert '\x1b' not in captured.err
    records = [json.loads(line) for line in structured.read_text().splitlines()]
    assert [r['level'] for r in records] == ['debug', 'warning']
    assert 'Internal detail' in readable.read_text()


def test_heartbeat_reports_blocking_work_and_stops_on_failure(tmp_path):
    path = tmp_path / 'events.jsonl'
    logs.configure(events_file=path, interval=0.02)
    observed = threading.Event()
    original = logs.emit
    def record(event):
        original(event)
        if 'still working' in event['message']:
            observed.set()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(logs, 'emit', record)
        with pytest.raises(RuntimeError, match='test failure'):
            with logs.Progress('Classifying', stage='safety', total=520, completed=19, prompt_id='adv_20') as progress:
                assert observed.wait(2), 'No heartbeat while operation was blocked'
                raise RuntimeError('test failure')
        assert not progress.thread.is_alive()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    heartbeats = [r for r in records if 'still working' in r['message']]
    assert heartbeats[0]['completed'] == 19 and heartbeats[0]['prompt_id'] == 'adv_20'
    assert records[-1]['level'] == 'error'
    assert not any(r['message'].endswith('finished') for r in records)


def test_inference_progress_counts_partial_resume_once(tmp_path):
    path = tmp_path / 'events.jsonl'
    logs.configure(events_file=path)
    rows = [{'workload': w, 'prompt_id': str(i), 'input_ids': [1, 2]} for w in ['math', 'chat'] for i in range(2)]
    with logs.InferenceProgress(rows, {('math', '0'): {}}, {'setup_id': 'my_gpu'}) as progress:
        progress.batch(rows[:2], 0)
        progress.saved(rows[:2], 1.5)
        progress.batch(rows[2:], 1)
        progress.saved(rows[2:], 0.2)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]['completed'] == 1 and records[0]['reused'] == 1
    assert records[-1]['completed'] == 4 and records[-1]['total'] == 4
    assert records[-1]['workload_saved'] == '2/2'
    assert {r.get('workload') for r in records} >= {'math', 'chat'}
    assert all(r['setting'] == 'my_gpu' for r in records)


def test_follower_skips_history_and_waits_for_complete_lines(tmp_path, capsys):
    logs.configure()
    path = tmp_path / 'child.events.jsonl'
    record = {'timestamp': '2026-09-10T12:00:00+00:00', 'level': 'info', 'stage': 'inference', 'message': 'old'}
    path.write_text(json.dumps(record) + '\n')
    follower = logs.EventFollower(path)
    record['message'] = 'new'
    data = json.dumps(record).encode()
    with path.open('ab') as handle:
        handle.write(data[:20])
    follower.drain()
    assert capsys.readouterr().err == ''
    with path.open('ab') as handle:
        handle.write(data[20:] + b'\n')
    follower.drain()
    follower.drain()
    output = capsys.readouterr().err
    assert 'old' not in output and output.count('new') == 1


def test_real_subprocess_forwards_live_events_and_keeps_raw_diagnostics(tmp_path, capsys):
    logs.configure(events_file=tmp_path / 'parent.jsonl', interval=0.05)
    command = [sys.executable, '-c', '''
import os, time
from driftbench_runner.logging import configure, event
configure(events_file=os.environ['DRIFTBENCH_EVENTS_FILE'], interval=.05)
print('RAW FRAMEWORK DIAGNOSTICS', flush=True)
event('Batch saved', stage='inference', workload='math', completed=2, total=4)
time.sleep(.3)
raise SystemExit(7)
''']
    with pytest.raises(RuntimeError, match='exited 7'):
        run_command(command, tmp_path / 'inference.log', {**os.environ, 'DRIFTBENCH_LOG_SETTING': 'gpu_test'})
    output = capsys.readouterr().err
    assert output.count('inference/math | Batch saved') == 1
    assert '[gpu_test]' in output and '2/4 (50.0%)' in output
    assert 'RAW FRAMEWORK DIAGNOSTICS' not in output
    assert 'RAW FRAMEWORK DIAGNOSTICS' in (tmp_path / 'inference.log').read_text()
    assert 'Batch saved' in (tmp_path / 'parent.jsonl').read_text()


@pytest.mark.parametrize('options', [[], ['--log-level', 'warning'], ['--progress-interval', '.1']])
def test_dry_run_stdout_remains_json_without_output_directory(tmp_path, options):
    directory = tmp_path / 'dry'
    result = subprocess.run([sys.executable, '-m', 'driftbench_runner', 'suite', '--config',
                             str(ROOT / 'suites/rtx3060.json'), '--output', str(directory), '--dry-run', *options],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['expected_records_per_setting'] == 2284
    assert not directory.exists()


def test_cli_error_is_readable_and_detailed_trace_is_saved(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'driftbench_runner', 'collect', str(tmp_path / 'missing'),
                             '--output', str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode == 1 and result.stdout == ''
    assert 'ERROR' in result.stderr and 'Traceback' not in result.stderr
    assert '\n    Traceback' in (tmp_path / 'out/logs/collect.log').read_text()


def test_status_works_before_manifest_and_for_copied_partial_suite(tmp_path):
    write_json(tmp_path / 'suite.json', {'status': 'running', 'plan': {
        'sources': {'math': {'selected': 3}}, 'settings': [{'id': 'a', 'run_dir': '/old/host/a'}, {'id': 'b'}]},
        'settings': {'a': {'status': 'inference_running'}, 'b': {'status': 'pending'}}})
    directory = tmp_path / 'settings/a'
    directory.mkdir(parents=True)
    (directory / 'math.jsonl').write_bytes(b'{"saved": 1}\n{"partial":')
    report = snapshot(tmp_path)
    assert report['settings']['a']['workloads']['math']['saved'] == 1
    assert report['settings']['b']['workloads']['math']['saved'] == 0
    assert 'a: inference_running | 1/3' in format_snapshot(report)
    assert 'b: pending | 0/3' in format_snapshot(report)
