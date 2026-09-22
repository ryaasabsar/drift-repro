"""Model-free, stage-level gated RMSNorm correctness benchmark."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from .common import digest, now, read_json, write_json
from .numeric import require
from .data import (FIXTURE, STAGES, PARENTS, compare_values, decode_bits,
                             load_fixture, logical_bits, make_fixture, quantize)


def implementation_hash():
    return digest({p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))})


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal(path, report):
    report['fingerprint'] = digest({k: v for k, v in report.items() if k != 'fingerprint'})
    write_json(path, report)


def load_result(directory):
    path = Path(directory)
    meta = read_json(path / 'result.json')
    require(meta['status'] == 'complete', f'Incomplete worker: {path}')
    require(digest({k: v for k, v in meta.items() if k != 'fingerprint'}) == meta['fingerprint'], 'Changed result metadata')
    require(digest((path / 'outputs.npz').read_bytes()) == meta['arrays_sha256'], 'Changed output tensors')
    arrays = dict(np.load(path / 'outputs.npz', allow_pickle=False))
    for key, record in meta['records'].items():
        bits = arrays[key]
        require(list(bits.shape) == record['shape'], 'Output shape changed')
        require(bits.dtype == (np.uint16 if record['dtype'] == 'bfloat16' else np.uint32), 'Output bit storage changed')
        require(digest(bits.tobytes()) == record['repeat_sha256'][-1], 'Output does not match recorded final repetition')
    return meta, arrays


def worker(plan_path):
    from .backends import TTBackend, TorchBackend
    plan = read_json(plan_path)
    output = Path(plan['output'])
    fixture, data = load_fixture(plan['fixture'])
    require(fixture['fingerprint'] == plan['protocol']['fixture_fingerprint'], 'Fixture changed after planning')
    require(implementation_hash() == plan['protocol']['implementation_sha256'], 'Code changed after planning')
    result = {'schema_version': 1, 'status': 'running', 'started_at': now(), 'protocol': plan['protocol'],
              'records': {}, 'controls': {}, 'errors': []}
    backend = None
    stored = {}
    def save():
        seal(output / 'result.json', result)
    save()
    try:
        backend = TTBackend(plan['tt_device_id']) if plan['backend'] == 'tenstorrent' else TorchBackend(
            plan['backend'], control=plan['protocol'].get('control', 'native'))
        result.update(environment=backend.info, native_control=backend.native_info,
                      instrumentation={'status': 'available' if plan['backend'] in ('nvidia', 'amd') else 'unavailable',
                                       'note': 'Triton diagnostic copy. Trust only when output matches native before/after checks.'})

        def upload(array, precision):
            tensor = backend.upload(array, precision)
            require(backend.dtype(tensor) == precision, 'Backend changed the requested input dtype')
            actual = backend.download(tensor)
            require(np.array_equal(logical_bits(actual, precision), logical_bits(array, precision)),
                    'Device upload changed logical input bits')
            return tensor

        def record(case, precision, mode, stage, tensor, reference, anchor=None):
            dtype = backend.dtype(tensor)
            expected_dtype = 'bfloat16' if stage == 'output' or mode == 'native' else precision
            require(dtype == expected_dtype, f'{mode}/{stage}: requested {expected_dtype}, got {dtype}; no silent downcast allowed')
            value = backend.download(tensor)
            require(value.shape == reference.shape, f'{mode}/{stage}: logical shape mismatch')
            bits = logical_bits(value, dtype)
            key = f'{case}/{precision}/{mode}/{stage}'
            entry = result['records'].setdefault(key, {'case': case, 'precision': precision, 'mode': mode,
                'stage': stage, 'dtype': dtype, 'shape': list(bits.shape), 'repeat_sha256': [],
                'operand_fingerprint': anchor, 'reference_errors': []})
            entry['repeat_sha256'].append(digest(bits.tobytes()))
            entry['reference_errors'].append(compare_values(value, reference, dtype))
            entry['repeat_bitwise_equal'] = len(set(entry['repeat_sha256'])) == 1
            stored[key] = bits.copy()
            return bits

        for case in fixture['cases']:
            name = case['name']
            if name not in plan['protocol']['cases']:
                continue
            print(f'Case {name}', flush=True)
            if case['kind'] == 'cast':
                anchor = data[f'{name}/input/precast']
                for _ in range(plan['protocol']['repeats']):
                    output_tensor = backend.operation('output', [upload(anchor, 'float32')], case['shape'][-1], case['eps'])
                    record(name, 'float32', 'isolated', 'output', output_tensor,
                           data[f'{name}/reference/output'], digest(anchor.tobytes()))
                save()
                continue
            inputs = {key: data[f'{name}/input/{key}'] for key in ('x', 'w', 'z')}
            width = case['shape'][-1]
            native_before = backend.native(inputs, case['eps'])
            before_bits = None if native_before is None else logical_bits(backend.download(native_before), 'bfloat16')
            control_matches, native_hashes = [], []
            for repeat in range(plan['protocol']['repeats']):
                if native_before is not None:
                    native = native_before if repeat == 0 else backend.native(inputs, case['eps'])
                    native_bits = record(name, 'float32', 'native', 'output', native, data[f'{name}/reference/output'])
                    native_hashes.append(digest(native_bits.tobytes()))
                trace = backend.instrument(inputs, case['eps'])
                if trace is not None:
                    for stage in STAGES:
                        bits = record(name, 'float32', 'instrumented', stage, trace[stage], data[f'{name}/reference/{stage}'])
                        if stage == 'output':
                            control_matches.append(bool(np.array_equal(bits, native_bits)))
                for precision in plan['protocol']['precisions']:
                    pipeline = {k: upload(quantize(v, precision), precision) for k, v in inputs.items()}
                    eps = float(quantize(case['eps'], precision))
                    for stage in STAGES:
                        pipeline[stage] = backend.operation(stage, [pipeline[p] for p in PARENTS[stage]], width, eps)
                        record(name, precision, 'pipeline', stage, pipeline[stage], data[f'{name}/reference/{stage}'])
                    # Each primitive is also run with fixed, shared operands.
                    # No output from this GPU becomes input to another isolated stage.
                    for stage in STAGES:
                        prefix = f'{name}/isolated/{precision}/{stage}'
                        anchors = [data[f'{prefix}/arg{i}'] for i in range(len(PARENTS[stage]))]
                        actual = backend.operation(stage, [upload(a, precision) for a in anchors], width, eps)
                        record(name, precision, 'isolated', stage, actual, data[f'{prefix}/reference'],
                               digest([digest(a.tobytes()) for a in anchors]))
                    del pipeline
            native_after = backend.native(inputs, case['eps'])
            result['controls'][name] = {
                'native_repeatable': len(set(native_hashes)) == 1 if native_hashes else None,
                'native_before_after_equal': None if native_after is None else bool(np.array_equal(
                    before_bits, logical_bits(backend.download(native_after), 'bfloat16'))),
                'instrumented_matches_native_each_repeat': control_matches,
                'instrumented_matches_native': all(control_matches) if control_matches else None,
                'note': 'An unequal control invalidates attribution of diagnostic intermediates to the original fused kernel.'}
            save()
        result['status'] = 'complete'
    except BaseException as exc:
        result['status'] = 'failed'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            np.savez_compressed(output / 'outputs.npz', **stored)
            result['arrays_sha256'] = digest((output / 'outputs.npz').read_bytes())
            result['finished_at'] = now()
            save()
    rows = []
    for entry in result['records'].values():
        rows.append({k: entry[k] for k in ('case', 'precision', 'mode', 'stage', 'dtype', 'repeat_bitwise_equal')} |
                    {k: entry['reference_errors'][-1][k] for k in ('differing_percent', 'max_absolute_error',
                       'relative_l2_error', 'ulp_max', 'ulp_p99', 'nonfinite_pairs')})
    csv_write(output / 'reference-errors.csv', rows)
    focus = []
    for key, entry in result['records'].items():
        if entry['case'] != 'captured':
            continue
        actual = decode_bits(stored[key], entry['dtype'])
        if entry['mode'] == 'isolated':
            ref_key = f"captured/isolated/{entry['precision']}/{entry['stage']}/reference"
        else:
            ref_key = f"captured/reference/{entry['stage']}"
        ref = data[ref_key]
        rounded = decode_bits(logical_bits(ref, entry['dtype']), entry['dtype'])
        for row, column in fixture['focus_positions']:
            position = (row, 0 if actual.shape[-1] == 1 else column)
            focus.append({k: entry[k] for k in ('case', 'precision', 'mode', 'stage', 'dtype')} |
                         {'focus_row': row, 'focus_column': column, 'actual': float(actual[position]),
                          'float64_reference': float(ref[position]), 'rounded_reference': float(rounded[position])})
    csv_write(output / 'focus-values.csv', focus)


def compare_workers(left, right):
    a, aa = load_result(left)
    b, bb = load_result(right)
    require(a['protocol'] == b['protocol'], 'Different fixtures, benchmark code, cases, precisions or repetition counts')
    rows = []
    examples = []
    for key in sorted(set(aa) & set(bb)):
        x, y = a['records'][key], b['records'][key]
        require(x['dtype'] == y['dtype'] and x['shape'] == y['shape'], 'Different output formats')
        require(x['operand_fingerprint'] == y['operand_fingerprint'], 'Different isolated operands')
        stats = compare_values(decode_bits(aa[key], x['dtype']), decode_bits(bb[key], y['dtype']), x['dtype'])
        trusted = None
        if x['mode'] == 'instrumented':
            controls = [meta['controls'][x['case']] for meta in (a, b)]
            trusted = all(c['instrumented_matches_native'] and c['native_before_after_equal'] and c['native_repeatable'] for c in controls)
        rows.append({k: x[k] for k in ('case', 'precision', 'mode', 'stage', 'dtype')} |
                    {'left_repeatable': x['repeat_bitwise_equal'], 'right_repeatable': y['repeat_bitwise_equal'],
                     'instrumentation_controls_pass': trusted, **stats})
        av, bv = decode_bits(aa[key], x['dtype']), decode_bits(bb[key], y['dtype'])
        for position in np.argwhere(aa[key] != bb[key])[:5]:
            index = tuple(position)
            left_value, right_value = float(av[index]), float(bv[index])
            examples.append({'case': x['case'], 'precision': x['precision'], 'mode': x['mode'],
                             'stage': x['stage'], 'index': position.tolist(),
                             'left_value': left_value if np.isfinite(left_value) else str(left_value),
                             'right_value': right_value if np.isfinite(right_value) else str(right_value),
                             'left_bits': int(aa[key][index]), 'right_bits': int(bb[key][index])})
    first = []
    for case in a['protocol']['cases']:
        for precision in a['protocol']['precisions']:
            for mode in ('pipeline', 'instrumented'):
                candidates = [r for r in rows if r['case'] == case and r['precision'] == precision and
                              r['mode'] == mode and not r['bitwise_equal']]
                if candidates:
                    row = min(candidates, key=lambda r: STAGES.index(r['stage']))
                    first.append({'case': case, 'precision': precision, 'mode': mode, 'stage': row['stage'],
                                  'instrumentation_controls_pass': row['instrumentation_controls_pass']})
    return {'left': str(left), 'right': str(right), 'left_environment': a['environment'],
            'right_environment': b['environment'], 'rows': rows, 'first_differing_stages': first,
            'mismatch_examples': examples,
            'isolated_differences': [r for r in rows if r['mode'] == 'isolated' and not r['bitwise_equal']],
            'missing_left': sorted(set(bb) - set(aa)), 'missing_right': sorted(set(aa) - set(bb)),
            'note': 'Identical operands isolate primitive output differences, not a hardware instruction or accuracy impact. Stage order is a dependency graph; sigmoid is a separate branch.'}


def compare_campaigns(left, right, output):
    left, right, output = Path(left), Path(right), Path(output)
    require(not output.exists(), 'Comparison exists; choose a new directory')
    a, b = read_json(left / 'campaign.json'), read_json(right / 'campaign.json')
    require(a['status'] == b['status'] == 'complete', 'Both campaigns must be complete')
    require(a['processes'] == b['processes'], 'Different process counts')
    output.mkdir(parents=True)
    rows, reports, examples = [], [], []
    for i in range(1, a['processes'] + 1):
        report = compare_workers(left / f'process-{i:03d}', right / f'process-{i:03d}')
        reports.append(report)
        write_json(output / f'process-{i:03d}.json', report)
        rows.extend({'process': i, **r} for r in report['rows'])
        examples.extend({'process': i, **r} for r in report['mismatch_examples'])
    csv_write(output / 'stages.csv', rows)
    csv_write(output / 'mismatch-examples.csv', examples)
    text = ['# Gated RMSNorm comparison', '',
            'Inspect [all stages](stages.csv). Nonempty mismatches also produce `mismatch-examples.csv` with values and raw bits. '
            'Bitwise differences are numerical observations, not automatic correctness failures.', '',
            '## First differing pipeline stages', '']
    for i, report in enumerate(reports, 1):
        for item in report['first_differing_stages']:
            text.append(f"- Process {i}, {item['case']}, {item['precision']}, {item['mode']}: **{item['stage']}**; instrument/control agreement: {item['instrumentation_controls_pass']}.")
    text += ['', '## Isolated primitive differences', '']
    for item in reports[0]['isolated_differences']:
        text.append(f"- {item['case']} / {item['precision']} / **{item['stage']}**: {item['differing_percent']:.6g}% differ; max ULP {item['ulp_max']}.")
    text += ['', 'Isolated stages use the same frozen input tensors on both devices. Pipeline stages may inherit upstream differences. '
             'An instrumented stage is attributable to the original fused calculation only when both controls pass; '
             'this remains an observational check, not proof of identical compiled instructions. '
             'Check each campaign\'s within-process and restart comparisons before attributing a difference to platforms.', '',
             f"Unavailable entries on left: {len(reports[0]['missing_left'])}; on right: {len(reports[0]['missing_right'])}. "
             'TTNN has no Triton instrumented control. Native-control implementations can differ across backends.']
    (output / 'README.md').write_text('\n'.join(text) + '\n')
    return {'output': str(output), 'rows': len(rows)}


def run(args):
    fixture, _ = load_fixture(args.fixture)
    require(args.repeats >= 2 and args.processes >= 2 and args.timeout > 0, 'Use at least two calls and two fresh processes, with a positive timeout')
    cases = args.cases or [c['name'] for c in fixture['cases']]
    require(len(set(cases)) == len(cases) and set(cases) <= {c['name'] for c in fixture['cases']}, 'Unknown/duplicate cases')
    require(len(set(args.precisions)) == len(args.precisions), 'Duplicate precisions')
    require(args.control != 'vllm' or args.backend in ('nvidia', 'amd'), 'The optional vLLM control requires NVIDIA or AMD')
    output = Path(args.output or Path.cwd() / 'results' / f'{args.backend}-{time.time_ns()}').resolve()
    require(not output.exists(), 'Output exists; select a new --output')
    protocol = {'fixture_fingerprint': fixture['fingerprint'], 'implementation_sha256': implementation_hash(),
                'cases': cases, 'precisions': args.precisions, 'repeats': args.repeats,
                'control': args.control, 'stages': list(STAGES)}
    plan = {'fixture': str(Path(args.fixture).resolve()), 'backend': args.backend,
            'tt_device_id': args.tt_device_id, 'protocol': protocol}
    return execute_plan(plan, output, args.processes, args.repeats, args.timeout, args.dry_run)


def execute_plan(plan, output, processes, repeats, timeout, dry_run=False):
    """Run either experiment in fresh, bounded worker processes."""
    require(processes >= 2 and repeats >= 2 and timeout > 0,
            'Use at least two calls and two fresh processes, with a positive timeout')
    require(not output.exists(), 'Output exists; select a new --output')
    if dry_run:
        return {'output': str(output), **plan}
    output.mkdir(parents=True)
    state = {'status': 'running', 'started_at': now(), 'processes': processes, 'protocol': plan['protocol'],
             'backend': plan['backend'], 'workers': []}
    write_json(output / 'campaign.json', state)
    old = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        cache = Path(os.environ.get('NORMBENCH_CACHE_DIR', str(Path.cwd() / '.cache/normbench'))).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / 'accelerator.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for i in range(1, processes + 1):
                directory = output / f'process-{i:03d}'
                directory.mkdir()
                write_json(directory / 'plan.json', {**plan, 'output': str(directory)})
                print(f'Process {i}/{processes}: {directory / "worker.log"}', flush=True)
                with (directory / 'worker.log').open('w') as log:
                    child = subprocess.Popen([sys.executable, '-u', '-m', 'normbench', '_worker',
                                              str(directory / 'plan.json')], stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    try:
                        deadline, last_notice = time.monotonic() + timeout, time.monotonic()
                        while child.poll() is None:
                            require(time.monotonic() < deadline, f'Worker timed out; see {directory / "worker.log"}')
                            if time.monotonic() - last_notice >= 15:
                                print(f'Process {i}: running; log {directory / "worker.log"}', flush=True)
                                last_notice = time.monotonic()
                            time.sleep(.5)
                        require(child.returncode == 0, f'Worker exited {child.returncode}; see {directory / "worker.log"}')
                        meta, _ = load_result(directory)
                        state['workers'].append({'directory': directory.name,
                                                 'repeatable': all(r['repeat_bitwise_equal'] for r in meta['records'].values())})
                        if 'attribution_controls_pass' in meta:
                            state['workers'][-1]['attribution_controls_pass'] = meta['attribution_controls_pass']
                    finally:
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=5)
                        except ProcessLookupError:
                            pass
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                write_json(output / 'campaign.json', state)
            for i in range(2, processes + 1):
                compared = compare_workers(output / 'process-001', output / f'process-{i:03d}')
                write_json(output / f'restart-{i:03d}.json', compared)
                csv_write(output / f'restart-{i:03d}.csv', compared['rows'])
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        state['finished_at'] = now()
        write_json(output / 'campaign.json', state)
    if plan['protocol'].get('experiment') == 'interventions-v1':
        (output / 'README.md').write_text(
            '# Gated RMSNorm interventions\n\n'
            'Inspect `process-*/controls.json` before interpreting effects. '
            '`reference-errors.csv` and `focus-values.csv` accompany raw `outputs.npz`. '
            '`kernels/` holds compiled code and `source/` the exact Python implementation.\n\n'
            'Copy this entire folder, then use `bash run.sh compare-interventions LEFT RIGHT --output DIR`. '
            'A completed run is not proof that observer or intervention controls passed.\n')
    else:
        (output / 'README.md').write_text(
            '# Gated RMSNorm benchmark\n\nCompleted without loading a model.\n\n'
            'Each `process-*/result.json` contains controls, per-stage repetition hashes and float64 reference errors. '
            '`reference-errors.csv` is the tabular view; `focus-values.csv` follows the three captured positions through every stage. '
            '`outputs.npz` contains raw FP32/BF16 output bits. '
            '`restart-*.json/csv` compares fresh processes.\n\n'
            'Copy this complete folder for cross-device comparison with `bash run.sh compare LEFT RIGHT --output DIR`. '
            'A completed run can contain numerical differences or failed observer controls; it is not an automatic correctness pass.\n')
    return {'status': state['status'], 'output': str(output), 'workers': state['workers']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--backend', choices=['nvidia', 'amd', 'tenstorrent', 'cpu'], required=True)
    p.add_argument('--fixture', default=str(FIXTURE))
    p.add_argument('--output')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--processes', type=int, default=2)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--tt-device-id', type=int, default=0)
    p.add_argument('--control', choices=['native', 'vllm'], default='native',
                   help='Native standalone kernels by default; optionally validate against installed vLLM on NVIDIA/AMD')
    p.add_argument('--precisions', nargs='+', choices=['float32', 'bfloat16'], default=['float32', 'bfloat16'])
    p.add_argument('--cases', nargs='+')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('compare')
    p.add_argument('left'); p.add_argument('right'); p.add_argument('--output', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--inputs', help='Optional numeric NPZ containing BF16-representable x, w, z; defaults to bundled inputs')
    p.add_argument('--epsilon', type=float, default=1e-6)
    p.add_argument('--output', required=True)
    p = sub.add_parser('_worker')
    p.add_argument('plan')
    from .interventions import add_arguments
    add_arguments(sub)
    args = parser.parse_args()
    if args.command == '_worker':
        if read_json(args.plan)['protocol'].get('experiment') == 'interventions-v1':
            from .interventions import worker as intervention_worker
            intervention_worker(args.plan)
        else:
            worker(args.plan)
        return
    if args.command == 'run':
        result = run(args)
    elif args.command == 'compare':
        result = compare_campaigns(args.left, args.right, args.output)
    elif args.command == 'prepare':
        result = make_fixture(args.output, args.inputs, args.epsilon)
    else:
        from .interventions import dispatch
        result = dispatch(args)
    import json
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
