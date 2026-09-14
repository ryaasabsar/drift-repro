"""Readable runner events, separate from framework diagnostics and result data."""
from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time

LOGGER = logging.getLogger('driftbench_runner')
LOGGER.propagate = False
_INTERVAL = 15.0
_LOCK = threading.RLock()


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s' if hours else f'{minutes}m {seconds:02d}s'


def progress_interval():
    return _INTERVAL


def clean(value):
    # Keep each event on one physical line, including unexpected exception text.
    return str(value).replace('\x1b', r'\x1b').replace('\r', r'\r').replace('\n', r'\n')


def render(event):
    parts = [f"[{event['timestamp'][11:19]}Z] {event['level'].upper():7}"]
    if event.get('setting'):
        parts.append(f"[{clean(event['setting'])}]")
    stage = event.get('stage', 'runner')
    if event.get('workload'):
        stage += '/' + event['workload']
    parts.append(f"{stage} | {clean(event['message'])}")
    if 'completed' in event and 'total' in event:
        done, total = event['completed'], event['total']
        parts.append(f'{done}/{total}' + (f' ({done / total:.1%})' if total else ''))
    elif 'total' in event:
        parts.append(f"total={event['total']}")
    elif 'completed' in event:
        parts.append(f"completed={event['completed']}")
    if 'elapsed_seconds' in event:
        parts.append('elapsed ' + duration(event['elapsed_seconds']))
    excluded = {'timestamp', 'level', 'setting', 'stage', 'workload', 'message', 'completed', 'total', 'elapsed_seconds', 'pid'}
    parts += [f'{key}={clean(value)}' for key, value in event.items() if key not in excluded and value is not None]
    return ' | '.join(parts)


class EventFormatter(logging.Formatter):
    def format(self, record):
        result = render({k: v for k, v in record.event.items() if k != 'traceback'})
        if record.event.get('traceback'):
            result += '\n' + '\n'.join('    ' + clean(line) for line in record.event['traceback'].splitlines())
        return result


class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.event, ensure_ascii=False, default=str)


def configure(level='INFO', log_file=None, events_file=None, interval=15):
    global _INTERVAL
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('progress interval must be greater than zero')
    _INTERVAL = float(interval)
    with _LOCK:
        for handler in LOGGER.handlers[:]:
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.propagate = False
        LOGGER.disabled = False
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level.upper())
        console.setFormatter(EventFormatter())
        LOGGER.addHandler(console)
        for path, formatter in ((log_file, EventFormatter()), (events_file, JSONFormatter())):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(path, encoding='utf-8')
                handler.setFormatter(formatter)
                handler.setLevel(logging.DEBUG)
                LOGGER.addHandler(handler)


def emit(event):
    """Also used by a suite to relay a child's original timestamp and context."""
    with _LOCK:
        if not LOGGER.handlers:
            configure()
        LOGGER.log(getattr(logging, event['level'].upper()), event['message'], extra={'event': event})


def event(message, *, stage=None, setting=None, level='info', **details):
    emit({'timestamp': datetime.now(timezone.utc).isoformat(), 'level': level,
          'pid': os.getpid(), 'setting': setting or os.environ.get('DRIFTBENCH_LOG_SETTING'),
          'stage': stage or os.environ.get('DRIFTBENCH_LOG_STAGE', 'runner'), 'message': message, **details})


class Progress:
    """Bounded progress output plus a heartbeat while a blocking operation runs."""
    def __init__(self, message, *, stage, setting=None, total=None, completed=0, **details):
        self.message, self.stage, self.setting = message, stage, setting
        self.details = details
        if total is not None:
            self.details.update(total=total, completed=completed)
        self.started = self.last_notice = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

    def report(self, message=None, level='info'):
        with self.lock:
            event(message or self.message, stage=self.stage, setting=self.setting, level=level,
                  elapsed_seconds=round(time.monotonic() - self.started, 1), **self.details)
            self.last_notice = time.monotonic()

    def update(self, completed=None, *, force=False, message=None, **details):
        with self.lock:
            changed = details.get('workload', self.details.get('workload')) != self.details.get('workload')
            self.details.update(details)
            if completed is not None:
                self.details['completed'] = completed
            due = time.monotonic() - self.last_notice >= _INTERVAL
            finished = completed is not None and completed == self.details.get('total')
        if force or changed or due or finished:
            self.report(message)

    def _heartbeat(self):
        while not self.stop.wait(max(0.01, _INTERVAL - (time.monotonic() - self.last_notice))):
            if time.monotonic() - self.last_notice >= _INTERVAL:
                self.report(self.message + ' — still working')

    def __enter__(self):
        self.started = time.monotonic()
        self.report()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name='driftbench-progress')
        self.thread.start()
        return self

    def __exit__(self, kind, exc, tb):
        self.stop.set()
        if self.thread:
            self.thread.join()
        if kind:
            interrupted = issubclass(kind, KeyboardInterrupt)
            self.report(('Interrupted: ' if interrupted else 'Failed: ') + self.message + f' ({exc})',
                        'warning' if interrupted else 'error')
        else:
            self.report(self.message + ' — finished')


@contextmanager
def activity(message, *, stage, **details):
    with Progress(message, stage=stage, **details) as progress:
        yield progress


class EventFollower:
    """Read only new, complete JSONL events; resumed logs are never replayed."""
    def __init__(self, path, since=None):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if self.path.exists() and since is None else 0
        self.since = since
        self.last_event = None

    def drain(self):
        if not self.path.exists():
            return
        with self.path.open('rb') as handle:
            if self.path.stat().st_size < self.offset:
                self.offset = 0
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line.endswith(b'\n'):
                    break
                self.offset = handle.tell()
                try:
                    record = json.loads(line)
                    if not all(key in record for key in ('timestamp', 'level', 'stage', 'message')):
                        continue
                    if record['level'].upper() not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
                        continue
                    if self.since is not None and datetime.fromisoformat(record['timestamp']).timestamp() < self.since:
                        continue
                except (ValueError, UnicodeError):
                    continue
                emit(record)
                self.last_event = record


class InferenceProgress(Progress):
    def __init__(self, rows, existing, config):
        self.counts = Counter(row['workload'] for row in rows)
        self.saved_keys = set(existing)
        self.saved_counts = Counter(key[0] for key in existing)
        super().__init__('Generating responses', stage='inference', setting=config.get('setup_id'),
                         total=len(rows), completed=len(existing), reused=len(existing))

    def batch(self, batch, index):
        workload = batch[0]['workload']
        self.update(workload=workload, workload_saved=f'{self.saved_counts[workload]}/{self.counts[workload]}',
                    batch=index + 1, batch_size=len(batch), prompt_id=batch[0]['prompt_id'] if len(batch) == 1 else None,
                    max_input_tokens=max(len(row['input_ids']) for row in batch), batch_seconds=None)

    def saved(self, batch, elapsed):
        for row in batch:
            key = (row['workload'], row['prompt_id'])
            if key not in self.saved_keys:
                self.saved_keys.add(key)
                self.saved_counts[key[0]] += 1
        workload = batch[0]['workload']
        self.update(len(self.saved_keys), message='Responses saved',
                    force=self.saved_counts[workload] == self.counts[workload],
                    workload_saved=f'{self.saved_counts[workload]}/{self.counts[workload]}', batch_seconds=round(elapsed, 2))
