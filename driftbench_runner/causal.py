"""Serving-layer localization and portable, captured-operation microbenchmarks."""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from .common import ROOT, digest, read_json, write_json
from .causal_ops import (SUPPORTED, logical_identity, metrics, require, tensors,
                         validate_arrays, values)

COMMANDS = ('capture-serving', 'compare-serving', 'export-operation', 'replay-operation', 'compare-operation-replays')


def capture_plan(args):
    from .investigation import load_bundle
    root, bundle = load_bundle(args.bundle)
    requests = read_json(args.requests)
    require(requests['bundle_fingerprint'] == bundle['fingerprint'], 'Different probe bundle')
    require(digest({k: v for k, v in requests.items() if k != 'fingerprint'}) == requests['fingerprint'],
            'Probe manifest changed')
    require(args.model in bundle['models'], 'Model is not in this bundle')
    require(args.processes >= 1 and args.repeats >= 1, 'Positive processes/repeats required')
    require(args.context > 0 and args.kv_cache_mib > 0 and args.max_mib > 0, 'Positive context and byte budgets required')
    require(0 < args.gpu_memory_fraction < 1, 'GPU memory fraction must be between zero and one')
    require(args.timeout > 0, 'Positive timeout required')
    cases = [c for c in requests['cases'] if c['model_key'] == args.model and c['condition'] == args.condition
             and c['prompt_id'] in args.cases]
    require(len(args.cases) == len(set(args.cases)) and len(cases) == len(args.cases),
            'Every requested prompt must identify one probe in this model/condition')
    original = {(c['workload'], c['prompt_id']): c['request']['input_ids']
                for c in read_json(root / 'models' / args.model / 'cases.json')}
    for c in cases:
        prefix = original[(c['workload'], c['prompt_id'])]
        require(c['input_ids'][:len(prefix)] == prefix and len(c['input_ids']) == len(prefix) + c['position'],
                'Probe does not extend its frozen source input')
        require(digest(c['input_ids']) == c['prefix_sha256'], 'Probe tokens changed')
        require(len(c['input_ids']) + 1 <= args.context, 'Context must fit the complete prefix plus one output token')
    template = 'mi210' if args.device == 'mi210' else 'a100'
    config = deepcopy(read_json(root / 'models' / args.model / f'{template}.json'))
    config['hardware']['device'] = args.device
    engine = {'dtype': config['engine']['dtype'], 'max_model_len': args.context,
              'max_num_seqs': 1, 'max_num_batched_tokens': args.context,
              'gpu_memory_utilization': args.gpu_memory_fraction,
              'kv_cache_memory_bytes': args.kv_cache_mib * 1024**2,
              'tensor_parallel_size': 1, 'enforce_eager': True,
              'enable_prefix_caching': False, 'enable_chunked_prefill': True,
              'language_model_only': config['engine'].get('language_model_only', False)}
    return {'schema_version': 1, 'bundle_fingerprint': bundle['fingerprint'],
            'requests_fingerprint': requests['fingerprint'], 'config': config,
            'engine': engine, 'cases': cases, 'module': args.module,
            'max_bytes': args.max_mib * 1024**2, 'repeats': args.repeats,
            'note': 'Explicit diagnostic intervention: fixed cache bytes/context, serial eager full-prefix prefill.'}


def capture(args):
    from .logging import activity
    from .suite import stop_process
    plan = capture_plan(args)
    if args.dry_run:
        return plan
    python = Path(args.server_python).expanduser().absolute()  # Preserve virtualenv symlink.
    require(python.is_file() and os.access(python, os.X_OK), f'Missing serving Python: {python}')
    output = Path(args.output).resolve()
    require(not output.exists(), 'Capture output exists; retain it and choose a new directory')
    output.mkdir(parents=True)
    environment = {**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   'PYTHONUNBUFFERED': '1'}
    state = {'status': 'running', 'plan': plan, 'processes': []}
    write_json(output / 'campaign.json', state)
    old = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        (ROOT / 'results').mkdir(exist_ok=True)
        with (ROOT / 'results/.gpu-experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for repeat in range(args.processes):
                directory = output / f'process-{repeat + 1:03d}'
                directory.mkdir()
                local = {**plan, 'output': str(directory)}
                write_json(directory / 'plan.json', local)
                with (directory / 'worker.log').open('w') as log:
                    process = subprocess.Popen([str(python), '-m', 'driftbench_runner.causal_capture',
                                                str(directory / 'plan.json')], env=environment, cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        with activity('Capturing vLLM layer tensors', stage='capture', log=str(directory / 'worker.log')):
                            deadline = time.monotonic() + args.timeout
                            while process.poll() is None:
                                require(time.monotonic() < deadline, f'Capture timeout; see {directory / "worker.log"}')
                                time.sleep(1)
                        require(process.returncode == 0, f'Capture worker exited {process.returncode}; see {directory / "worker.log"}')
                        result = read_json(directory / 'capture-run.json')
                        require(result['status'] == 'complete', 'Incomplete worker capture')
                        state['processes'].append({'directory': directory.name, 'weights_unchanged': result['weights_unchanged'],
                                                   'observer_checks_equal': all(r['observer_check_equal'] for r in result['records'])})
                    finally:
                        stop_process(process)
                        # A parent that failed early may have left descendants
                        # in the process group we created for this worker.
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    write_json(output / 'campaign.json', state)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        write_json(output / 'campaign.json', state)
    return {'status': state['status'], 'processes': state['processes'], 'output': str(output)}


def load_trace(directory):
    directory = Path(directory)
    meta = read_json(directory / 'trace.json')
    require(meta.get('schema_version') == 1 and meta.get('status') == 'complete', 'Incomplete/unsupported trace')
    require(digest({k: v for k, v in meta.items() if k != 'fingerprint'}) == meta['fingerprint'], 'Trace manifest changed')
    require(digest((directory / 'tensors.npz').read_bytes()) == meta['tensors_sha256'], 'Trace arrays changed')
    arrays = dict(np.load(directory / 'tensors.npz', allow_pickle=False))
    validate_arrays(meta['events'], arrays)
    return meta, arrays


def compare(left, right, output):
    from .investigation import csv_file
    output = Path(output)
    require(not output.exists(), 'Comparison output exists')
    a, aa = load_trace(left)
    b, bb = load_trace(right)
    for key in ('model', 'revision', 'dtype', 'seed', 'engine', 'case', 'sampling', 'implementation_sha256'):
        require(a['protocol'][key] == b['protocol'][key], f'Incomparable captures: {key}')
    require(a['target_module'] == b['target_module'] and a['execution'] == b['execution'], 'Different capture coverage')
    shape = lambda r: (r['kind'], r['name'], r.get('call'), r.get('schema'))
    require([shape(r) for r in a['events']] == [shape(r) for r in b['events']],
            'Execution event sequences differ; inspect traces, do not pair by position')
    rows = []
    for x, y in zip(a['events'], b['events']):
        for side in ('inputs', 'args', 'kwargs', 'outputs'):
            require((side in x) == (side in y), 'Trace structure mismatch')
            if side not in x:
                continue
            at, bt = list(tensors(x[side])), list(tensors(y[side]))
            require(len(at) == len(bt), 'Tensor structure differs')
            for i, (u, v) in enumerate(zip(at, bt)):
                require(u['shape'] == v['shape'] and u['dtype'] == v['dtype'], 'Tensor shape/dtype differs')
                eq = np.array_equal(aa[u['tensor']], bb[v['tensor']])
                rows.append({'event': x['event'], 'kind': x['kind'], 'name': x['name'], 'side': side, 'tensor': i,
                             'bitwise_equal': bool(eq), 'layout_equal': u['stride'] == v['stride'],
                             **metrics(values(u, aa), values(v, bb))})
    changed = [r for r in rows if not r['bitwise_equal']]
    first_output = next((r for r in changed if r['side'] == 'outputs'), None)
    candidates = []
    for x, y in zip(a['events'], b['events']):
        if x['kind'] != 'operation' or 'args' not in x or 'args' not in y:
            continue
        same_input = (logical_identity(x['args']) == logical_identity(y['args']) and
                      logical_identity(x['kwargs']) == logical_identity(y['kwargs']))
        output_changed = any(r['event'] == x['event'] and r['side'] == 'outputs' and not r['bitwise_equal'] for r in rows)
        if output_changed:
            candidates.append({'event': x['event'], 'operation': x['name'], 'inputs_identical': same_input,
                               'replayable': x['replayable'] and y['replayable'],
                               'interpretation': 'Same explicit operands, different output; replay candidate, not proof of a hardware defect.'
                               if same_input else 'Input divergence is already upstream; this operation is not isolated.'})
    result = {'left': str(left), 'right': str(right), 'first_different_output': first_output,
              'different_tensor_count': len(changed), 'tensor_count': len(rows), 'operation_candidates': candidates,
              'weights_equal': a['protocol']['weights_sha256'] == b['protocol']['weights_sha256'],
              'left_environment': a['environment'], 'right_environment': b['environment'],
              'limitations': ['First observed boundary, not necessarily the first differing instruction.',
                             'Same explicit operands do not rule out hidden state in opaque custom kernels.',
                             'Review capture-run.json observer checks and parameter hashes before attributing a cause.']}
    output.mkdir(parents=True)
    csv_file(output / 'tensors.csv', rows)
    write_json(output / 'comparison.json', result)
    first = 'No differing output observed.' if first_output is None else (
        f"First differing output: event {first_output['event']}, `{first_output['name']}`.")
    (output / 'README.md').write_text(f'{first}\n\nInspect [all tensors](tensors.csv) and [operation candidates](comparison.json). '
                                    'A changed layer output localizes a boundary; it does not by itself prove a kernel bug.\n')
    return result


def export_operation(trace, event_id, output):
    root = Path(output)
    require(not root.exists(), 'Operation output exists')
    meta, arrays = load_trace(trace)
    matches = [e for e in meta['events'] if e['event'] == event_id]
    require(len(matches) == 1, 'Unknown event ID')
    event = matches[0]
    require(event['kind'] == 'operation' and event.get('replayable') and event['name'] in SUPPORTED,
            'This event needs a dedicated replay/reference adapter; no generic fallback is valid')
    needed = {t['tensor'] for t in tensors(event)}
    require(needed, 'Operation has no captured tensors')
    root.mkdir(parents=True)
    np.savez(root / 'tensors.npz', **{k: arrays[k] for k in sorted(needed)})
    operation = {'schema_version': 1, 'operation': event['name'], 'event': event,
                 'trace_fingerprint': meta['fingerprint'], 'protocol': meta['protocol'],
                 'capture_environment': meta['environment'], 'capture_execution': meta['execution'],
                 'tensors_sha256': digest((root / 'tensors.npz').read_bytes())}
    operation['fingerprint'] = digest(operation)
    write_json(root / 'operation.json', operation)
    (root / 'README.md').write_text(
        f"Captured operation: `{event['name']}`, event {event_id}.\n\n"
        'Copy this complete directory to either GPU machine. It contains numeric operands and outputs; '
        'no model download or original run directory is needed. Run from the repository root:\n\n'
        '```bash\nDRIFTBENCH_PYTHON=.venv/bin/python bash scripts/investigate.sh replay-operation \\\n'
        '  --bundle /path/to/this/directory --device cuda --repeats 3 --profile \\\n'
        '  --output results/operation-replay\n```\n\n'
        'On MI210 use `.venv-rocm-vllm/bin/python`. Set `--atol` and `--rtol` only with a justified '
        'numerical acceptance criterion; defaults request exact numeric agreement. '
        'The float64 reference starts from the captured quantized inputs. It is a numerical reference, '
        'not proof of the exact mathematical result or a promise of identical reduction order. '
        'Inspect `replay.json` and optional `kernel-trace.json`. No timing claim is made.\n')
    return {'operation': event['name'], 'event': event_id, 'output': str(root)}


def compare_replays(left, right, output):
    require(not Path(output).exists(), 'Replay comparison output exists')
    loaded = []
    for path in (Path(left), Path(right)):
        meta = read_json(path / 'replay.json')
        require(digest((path / 'outputs.npz').read_bytes()) == meta['outputs_sha256'], 'Replay outputs changed')
        loaded.append((meta, dict(np.load(path / 'outputs.npz', allow_pickle=False))))
    (a, aa), (b, bb) = loaded
    for key in ('operation_fingerprint', 'operation', 'implementation_sha256'):
        require(a[key] == b[key], f'Incomparable operation replays: {key}')
    require(aa.keys() == bb.keys(), 'Replay output structures differ')
    rows = []
    for key in sorted(k for k in aa if k.startswith('output_') and not k.endswith('__bits')):
        rows.append({'output': key, 'bitwise_equal': bool(np.array_equal(aa[key + '__bits'], bb[key + '__bits'])),
                     **metrics(aa[key], bb[key])})
    result = {'operation': a['operation'], 'operation_fingerprint': a['operation_fingerprint'],
              'left_environment': a['environment'], 'right_environment': b['environment'], 'outputs': rows,
              'left_repeatable': a['repeat_bitwise_equal'], 'right_repeatable': b['repeat_bitwise_equal'],
              'note': 'Same captured operands. Inspect each replay reference error; unequal bits alone are not a correctness verdict.'}
    write_json(output, result)
    return result


def add_parsers(sub):
    p = sub.add_parser('capture-serving', help='Capture real vLLM layer outputs and optional operation operands')
    for name in ('bundle', 'requests', 'model', 'server-python', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', choices=['a100', 'mi210', 'rtx3060'], required=True)
    p.add_argument('--condition', default='serial')
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--processes', type=int, default=2)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--module', help='Exact module name from modules.json; enables operation capture within it')
    p.add_argument('--max-mib', type=int, default=512)
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--kv-cache-mib', type=int, default=512)
    p.add_argument('--gpu-memory-fraction', type=float, default=.55)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('compare-serving', help='Locate the first differing captured serving-layer output')
    p.add_argument('left')
    p.add_argument('right')
    p.add_argument('--output', required=True)
    p = sub.add_parser('export-operation', help='Export actual captured operands into a portable microbenchmark')
    p.add_argument('--trace', required=True)
    p.add_argument('--event', type=int, required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('replay-operation', help='Replay a captured production operation against CPU float64')
    p.add_argument('--bundle', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--atol', type=float, default=0.)
    p.add_argument('--rtol', type=float, default=0.)
    p.add_argument('--profile', action='store_true')
    p = sub.add_parser('compare-operation-replays', help='Compare replay outputs with verified identical operand bundles')
    p.add_argument('left')
    p.add_argument('right')
    p.add_argument('--output', required=True)


def dispatch(args):
    if args.command == 'capture-serving':
        return capture(args)
    if args.command == 'compare-serving':
        return compare(args.left, args.right, args.output)
    if args.command == 'export-operation':
        return export_operation(args.trace, args.event, args.output)
    if args.command == 'replay-operation':
        from .causal_ops import replay
        return replay(args.bundle, args.device, args.output, args.repeats, args.atol, args.rtol, args.profile)
    if args.command == 'compare-operation-replays':
        return compare_replays(args.left, args.right, args.output)
    raise ValueError(args.command)
