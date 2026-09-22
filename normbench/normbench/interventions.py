"""Portable, controlled sigmoid/rsqrt interventions for NVIDIA and AMD.

Client commands need NumPy only. Device imports are confined to the worker.
"""
from pathlib import Path
import json
import re
import time

import numpy as np

from .common import digest, now, read_json, write_json
from .data import FIXTURE, STAGES, decode_bits, logical_bits, compare_values, load_fixture, reference
from .numeric import require

BUNDLE = Path(__file__).parent / 'fixtures/a100-mi210-intervention'
EXPERIMENT = 'interventions-v1'
# (replace reciprocal square root, replace sigmoid)
VARIANTS = {'original': (False, False), 'shared_sigmoid': (False, True),
            'shared_rsqrt': (True, False), 'shared_both': (True, True),
            'replay_original': (False, False),
            'self_sigmoid': (False, True), 'self_rsqrt': (True, False), 'self_both': (True, True)}
PRIMARY = tuple(list(VARIANTS)[:4])


def key(variant, stage='output', native=False):
    return f'captured/float32/{variant}{"-native" if native else ""}/{stage}'


def prepare_bundle(left, right, output, fixture=FIXTURE):
    from .cli import load_result, seal
    output = Path(output)
    require(not output.exists(), 'Bundle destination exists')
    fm, values = load_fixture(fixture)
    cases = [c for c in fm['cases'] if c['name'] == 'captured' and c['kind'] == 'norm']
    require(len(cases) == 1, 'Fixture must contain the captured normalization case')
    arrays, sources, protocols = {}, {}, []
    for directory in (Path(left), Path(right)):
        campaign = read_json(directory / 'campaign.json')
        require(campaign['status'] == 'complete' and campaign['processes'] >= 2, 'Need complete repeated source campaigns')
        first = None
        fingerprints = []
        for i in range(1, campaign['processes'] + 1):
            meta, bits = load_result(directory / f'process-{i:03d}')
            require(meta['protocol'] == campaign['protocol'], 'Source worker protocol differs from campaign')
            require(all(meta['controls']['captured'][k] for k in (
                'native_repeatable', 'native_before_after_equal', 'instrumented_matches_native')), 'Source observer controls failed')
            required = [f'captured/float32/instrumented/{s}' for s in STAGES] + ['captured/float32/native/output']
            require(all(meta['records'][k]['repeat_bitwise_equal'] for k in required), 'Source outputs are not repeatable')
            require(all(np.isfinite(decode_bits(bits[k], meta['records'][k]['dtype'])).all() for k in required), 'Nonfinite source output')
            if first is None:
                first = (meta, bits)
            else:
                require(all(np.array_equal(bits[k], first[1][k]) for k in required), 'Source changes across fresh processes')
            fingerprints.append(meta['fingerprint'])
        meta, bits = first
        backend = meta['environment']['backend']
        require(backend in ('nvidia', 'amd') and backend not in sources, 'Supply one NVIDIA and one AMD campaign')
        require(meta['protocol']['fixture_fingerprint'] == fm['fingerprint'], 'Source uses a different fixture')
        require(meta['native_control']['implementation'] == 'NormBench fused Triton kernel, snapshots disabled',
                'Source must use the original standalone native control')
        protocols.append(meta['protocol'])
        sources[backend] = {'environment': meta['environment'], 'worker_fingerprints': fingerprints,
                            'implementation_sha256': meta['protocol']['implementation_sha256']}
        for stage in STAGES:
            arrays[f'{backend}/trace/{stage}'] = bits[f'captured/float32/instrumented/{stage}']
        arrays[f'{backend}/native'] = bits['captured/float32/native/output']
    require(protocols[0] == protocols[1], 'Source protocols differ')
    # Do not compute rsqrt from an idealized float64 reduction: freeze the actual,
    # common FP32 operand consumed by the observed fused implementations.
    require(np.array_equal(arrays['nvidia/trace/add_epsilon'], arrays['amd/trace/add_epsilon']),
            'Actual rsqrt operands differ; cannot isolate rsqrt with this bundle')
    for name in ('x', 'w', 'z'):
        arrays[name] = values[f'captured/input/{name}']
    arrays['rsqrt_input'] = decode_bits(arrays['nvidia/trace/add_epsilon'], 'float32').copy()
    require(np.all(arrays['rsqrt_input'] > 0), 'rsqrt operands must be positive')
    arrays['shared_rsqrt'] = reference('rsqrt', [arrays['rsqrt_input']], cases[0]['shape'][1], cases[0]['eps']).astype(np.float32)
    arrays['shared_sigmoid'] = reference('sigmoid', [arrays['z']], cases[0]['shape'][1], cases[0]['eps']).astype(np.float32)
    for stage in STAGES:
        arrays[f'reference/{stage}'] = values[f'captured/reference/{stage}']
    output.mkdir(parents=True)
    np.savez_compressed(output / 'inputs.npz', **arrays)
    manifest = {'schema_version': 1, 'experiment': EXPERIMENT, 'fixture_fingerprint': fm['fingerprint'],
                'case': cases[0], 'focus_positions': fm['focus_positions'], 'sources': sources,
                'source_protocol': protocols[0], 'arrays_sha256': digest((output / 'inputs.npz').read_bytes()),
                'reference_method': 'NumPy float64 rsqrt/sigmoid of exact frozen operands, rounded once to FP32; not an exact-real oracle.',
                'baseline_differing_elements': int(np.count_nonzero(arrays['nvidia/native'] != arrays['amd/native']))}
    seal(output / 'bundle.json', manifest)
    return {'output': str(output), 'fingerprint': manifest['fingerprint'],
            'baseline_differing_elements': manifest['baseline_differing_elements']}


def load_bundle(path=BUNDLE):
    path = Path(path)
    m = read_json(path / 'bundle.json')
    require(m['schema_version'] == 1 and m['experiment'] == EXPERIMENT, 'Unsupported intervention bundle')
    require(digest({k: v for k, v in m.items() if k != 'fingerprint'}) == m['fingerprint'], 'Changed bundle metadata')
    require(digest((path / 'inputs.npz').read_bytes()) == m['arrays_sha256'], 'Changed bundle tensors')
    with np.load(path / 'inputs.npz', allow_pickle=False) as archive:
        data = dict(archive)
    rows, columns = m['case']['shape']
    for name, shape in {'x': (rows, columns), 'z': (rows, columns), 'w': (1, columns),
                        'rsqrt_input': (rows, 1), 'shared_rsqrt': (rows, 1), 'shared_sigmoid': (rows, columns)}.items():
        require(data[name].dtype == np.float32 and data[name].shape == shape and np.isfinite(data[name]).all(),
                f'Invalid bundle array: {name}')
    require(np.array_equal(logical_bits(data['rsqrt_input'], 'float32'), data['nvidia/trace/add_epsilon']) and
            np.array_equal(data['nvidia/trace/add_epsilon'], data['amd/trace/add_epsilon']), 'Bundle rsqrt operands are not shared')
    return m, data


def save_kernel(kernel, folder, geometry, backend):
    """Save actual compiled artifacts; never claim PTX is NVIDIA machine assembly."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    formats = ('ttir', 'ttgir', 'llir', 'ptx', 'cubin') if backend == 'nvidia' else ('ttir', 'ttgir', 'llir', 'amdgcn', 'hsaco')
    files, excerpts = {}, {}
    for fmt in formats:
        require(fmt in kernel.asm, f'Missing compiled {fmt}; disable binary-only Triton caching and retry with a fresh cache')
        content = kernel.asm[fmt]
        raw = content.encode() if isinstance(content, str) else bytes(content)
        name = f'kernel.{fmt}'
        (folder / name).write_bytes(raw)
        files[name] = digest(raw)
        if fmt in ('llir', 'ptx', 'amdgcn'):
            excerpts[fmt] = [line.strip() for line in content.splitlines()
                             if re.search(r'rsqrt|sqrt|exp2?|rcp|div|fma|cvt', line, re.I)]
    metadata = json.loads(json.dumps(kernel.metadata._asdict(), default=str))
    result = {'files': files, 'metadata': metadata, 'launch': geometry,
              'binary_sha256': files['kernel.cubin' if backend == 'nvidia' else 'kernel.hsaco'],
              'instruction_excerpts': excerpts,
              'note': 'Instruction matches are search aids, not automatic root-cause attribution. PTX is virtual ISA; cubin is retained for optional disassembly.'}
    write_json(folder / 'manifest.json', result)
    return result


def unaffected_stages(replace_rsqrt, replace_sigmoid):
    stages = ['square', 'sum_squares', 'mean_square', 'add_epsilon']
    if not replace_rsqrt:
        stages += ['rsqrt', 'normalized', 'weighted']
    if not replace_sigmoid:
        stages += ['sigmoid', 'gate']
    return stages


def worker(plan_path):
    from .backends import TorchBackend
    from .intervention_kernel import launch
    from .cli import implementation_hash, seal, csv_write
    plan = read_json(plan_path)
    out = Path(plan['output'])
    bm, data = load_bundle(plan['bundle'])
    require(bm['fingerprint'] == plan['protocol']['bundle_fingerprint'], 'Bundle changed after planning')
    require(implementation_hash() == plan['protocol']['implementation_sha256'], 'Code changed after planning')
    backend = None
    stored = {}
    result = {'schema_version': 1, 'status': 'running', 'started_at': now(), 'protocol': plan['protocol'],
              'records': {}, 'errors': [], 'kernel_artifacts': {}, 'controls': {}}
    checks = {v: {'capture_matches_native': [], 'unaffected_stages_equal': [], 'injected_values_equal': []} for v in VARIANTS}
    baseline_checks, operand_checks, control_differences = [], [], []
    def save():
        seal(out / 'result.json', result)
    def bits_of(tensor):
        dtype = backend.dtype(tensor)
        values = backend.download(tensor)
        require(np.isfinite(values).all(), 'Nonfinite intervention output')
        return logical_bits(values, dtype)
    def check_trace(label, actual, expected, repeat):
        equal = True
        for stage in STAGES:
            if np.array_equal(actual[stage], expected[stage]):
                continue
            equal = False
            dtype = 'bfloat16' if stage == 'output' else 'float32'
            stats = compare_values(decode_bits(actual[stage], dtype), decode_bits(expected[stage], dtype), dtype)
            control_differences.append({'repeat': repeat + 1, 'control': label, 'stage': stage,
                **{k: stats[k] for k in ('differing_elements', 'elements', 'differing_percent', 'ulp_max')},
                'first_coordinates': json.dumps(np.argwhere(actual[stage] != expected[stage])[:5].tolist())})
        return equal
    def record(variant, stage, tensor, native=False):
        dtype = 'bfloat16' if stage == 'output' else 'float32'
        require(backend.dtype(tensor) == dtype, 'Intervention changed output dtype')
        values = backend.download(tensor)
        require(np.isfinite(values).all() and values.shape == data[f'reference/{stage}'].shape, 'Invalid intervention output')
        bits = logical_bits(values, dtype)
        k = key(variant, stage, native)
        r = result['records'].setdefault(k, {'case': 'captured', 'precision': 'float32',
            'mode': f'{variant}{"-native" if native else ""}', 'stage': stage, 'dtype': dtype,
            'shape': list(bits.shape), 'repeat_sha256': [], 'reference_errors': [], 'operand_fingerprint': bm['fingerprint']})
        r['repeat_sha256'].append(digest(bits.tobytes()))
        r['repeat_bitwise_equal'] = len(set(r['repeat_sha256'])) == 1
        r['reference_errors'].append(compare_values(values, data[f'reference/{stage}'], dtype))
        stored[k] = bits.copy()
        return bits
    def run_variant(name, replacements, capture):
        tensors, compiled, geometry = launch(inputs, replacements, bm['case']['eps'], capture, switches[name])
        tag = f'{name}-{"trace" if capture else "native"}'
        if tag not in result['kernel_artifacts']:
            result['kernel_artifacts'][tag] = save_kernel(compiled, out / 'kernels' / tag, geometry, plan['backend'])
        return tensors
    save()
    try:
        backend = TorchBackend(plan['backend'])
        result.update(environment=backend.info, native_control=backend.native_info,
                      source_environment=bm['sources'][plan['backend']]['environment'])
        (out / 'source').mkdir()
        result['source_files'] = {}
        for path in sorted(Path(__file__).parent.glob('*.py')):
            (out / 'source' / path.name).write_bytes(path.read_bytes())
            result['source_files'][path.name] = digest(path.read_bytes())
        def upload(value, dtype):
            tensor = backend.upload(value, dtype)
            require(backend.dtype(tensor) == dtype and np.array_equal(bits_of(tensor), logical_bits(value, dtype)), 'Device upload changed operand bits')
            return tensor
        inputs = {name: upload(data[name], 'bfloat16') for name in ('x', 'w', 'z')}
        shared = {name: upload(data[f'shared_{name}'], 'float32') for name in ('rsqrt', 'sigmoid')}
        switches = {name: backend.torch.tensor(flags, dtype=backend.torch.int32, device='cuda')
                    for name, flags in VARIANTS.items()}
        switches['original'] = None
        before = bits_of(backend.native({k: data[k] for k in inputs}, bm['case']['eps']))
        for repeat in range(plan['protocol']['repeats']):
            print(f'Intervention repetition {repeat + 1}', flush=True)
            original_native = run_variant('original', shared, False)
            original_trace = run_variant('original', shared, True)
            original_bits = {s: bits_of(t) for s, t in original_trace.items()}
            native_bits = bits_of(original_native['output'])
            baseline_checks.append(bool(np.array_equal(native_bits, data[f'{plan["backend"]}/native']) and
                all(np.array_equal(original_bits[s], data[f'{plan["backend"]}/trace/{s}']) for s in STAGES)))
            operand_checks.append(bool(np.array_equal(original_bits['add_epsilon'], logical_bits(data['rsqrt_input'], 'float32'))))
            replay_native = run_variant('replay_original', shared, False)
            replay_trace = run_variant('replay_original', shared, True)
            replay_bits = {s: bits_of(t) for s, t in replay_trace.items()}
            checks['replay_original'].setdefault('legacy_native_equal', []).append(bool(
                np.array_equal(bits_of(replay_native['output']), native_bits)))
            checks['replay_original'].setdefault('legacy_trace_equal', []).append(
                check_trace('replay_original_vs_legacy', replay_bits, original_bits, repeat))
            # Self controls replay the values computed by this exact diagnostic
            # binary. The separate legacy bridge is checked and exposed below.
            self_values = {s: replay_trace[s] for s in ('rsqrt', 'sigmoid')}
            for name, (rs, sg) in VARIANTS.items():
                replacements = self_values if name.startswith('self_') else shared
                native = original_native if name == 'original' else replay_native if name == 'replay_original' else run_variant(name, replacements, False)
                trace = original_trace if name == 'original' else replay_trace if name == 'replay_original' else run_variant(name, replacements, True)
                nb = record(name, 'output', native['output'], native=True)
                tb = {s: record(name, s, trace[s]) for s in STAGES}
                checks[name]['capture_matches_native'].append(bool(np.array_equal(nb, tb['output'])))
                baseline_bits = original_bits if name == 'original' else replay_bits
                checks[name]['unaffected_stages_equal'].append(all(np.array_equal(tb[s], baseline_bits[s]) for s in unaffected_stages(rs, sg)))
                checks[name]['injected_values_equal'].append(all(np.array_equal(tb[s], bits_of(replacements[s]))
                                                                for s, yes in (('rsqrt', rs), ('sigmoid', sg)) if yes))
                if name.startswith('self_'):
                    checks[name].setdefault('self_native_equal', []).append(bool(np.array_equal(nb, replay_bits['output'])))
                    checks[name].setdefault('self_trace_equal', []).append(check_trace(name, tb, replay_bits, repeat))
            save()
        after = bits_of(backend.native({k: data[k] for k in inputs}, bm['case']['eps']))
        global_checks = {'historical_baseline_reproduced': all(baseline_checks),
                         'original_operands_match_bundle': all(operand_checks),
                         'original_native_before_after_equal': bool(np.array_equal(before, after) and
                             np.array_equal(after, stored[key('original', native=True)]))}
        controls = {'global': global_checks, 'variants': {}}
        for name in VARIANTS:
            flags = {k: all(v) for k, v in checks[name].items()}
            flags['repeatable'] = all(r['repeat_bitwise_equal'] for r in result['records'].values()
                                     if r['mode'] in (name, name + '-native'))
            controls['variants'][name] = flags
        for name in PRIMARY:
            flags = controls['variants'][name]
            if name != 'original':
                placebo = name.replace('shared_', 'self_')
                flags['self_control_pass'] = all(controls['variants'][placebo].values())
                flags['self_and_shared_binary_equal'] = all(
                    result['kernel_artifacts'][f'{name}-{path}']['binary_sha256'] ==
                    result['kernel_artifacts'][f'{placebo}-{path}']['binary_sha256'] for path in ('native', 'trace'))
                flags['replay_original_control_pass'] = all(controls['variants']['replay_original'].values())
                flags['replay_and_shared_binary_equal'] = all(
                    result['kernel_artifacts'][f'{name}-{path}']['binary_sha256'] ==
                    result['kernel_artifacts'][f'replay_original-{path}']['binary_sha256'] for path in ('native', 'trace'))
            flags['attribution_eligible'] = all(global_checks.values()) and all(flags.values())
        result['controls'] = controls
        result['attribution_controls_pass'] = all(controls['variants'][v]['attribution_eligible'] for v in PRIMARY)
        result['control_differences'] = control_differences
        write_json(out / 'controls.json', controls)
        csv_write(out / 'control-differences.csv', control_differences)
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
            np.savez_compressed(out / 'outputs.npz', **stored)
            result['arrays_sha256'] = digest((out / 'outputs.npz').read_bytes())
            result['finished_at'] = now()
            save()
    errors, focus = [], []
    for k, r in result['records'].items():
        e = r['reference_errors'][-1]
        errors.append({'mode': r['mode'], 'stage': r['stage'], 'repeatable': r['repeat_bitwise_equal'],
                       **{field: e[field] for field in ('differing_percent', 'ulp_max', 'max_absolute_error', 'relative_l2_error')}})
        actual = decode_bits(stored[k], r['dtype'])
        for row, col in bm['focus_positions']:
            idx = (row, 0 if actual.shape[1] == 1 else col)
            focus.append({'mode': r['mode'], 'stage': r['stage'], 'row': row, 'column': col,
                          'value': float(actual[idx]), 'bits': int(stored[k][idx]),
                          'float64_reference': float(data[f'reference/{r["stage"]}'][idx])})
    csv_write(out / 'reference-errors.csv', errors)
    csv_write(out / 'focus-values.csv', focus)


def run(args):
    from .cli import implementation_hash, execute_plan
    bm, _ = load_bundle(args.bundle)
    protocol = {'experiment': EXPERIMENT, 'fixture_fingerprint': bm['fixture_fingerprint'],
                'bundle_fingerprint': bm['fingerprint'], 'implementation_sha256': implementation_hash(),
                'cases': ['captured'], 'precisions': ['float32'], 'repeats': args.repeats,
                'variants': list(VARIANTS), 'stages': list(STAGES)}
    plan = {'bundle': str(Path(args.bundle).resolve()), 'backend': args.backend, 'protocol': protocol}
    output = Path(args.output or Path.cwd() / 'results' / f'{args.backend}-interventions-{time.time_ns()}').resolve()
    return execute_plan(plan, output, args.processes, args.repeats, args.timeout, args.dry_run)


def effect_counts(original_left, original_right, changed_left, changed_right):
    require(original_left.shape == original_right.shape == changed_left.shape == changed_right.shape, 'Different effect shapes')
    before, after = original_left != original_right, changed_left != changed_right
    return {'baseline_differing': int(before.sum()), 'differing': int(after.sum()),
            'differing_percent': float(after.mean() * 100), 'resolved': int((before & ~after).sum()),
            'remaining': int((before & after).sum()), 'new': int((~before & after).sum()), 'elements': int(before.size)}


def verify_artifacts(directory, meta):
    directory = Path(directory)
    for tag, artifact in meta['kernel_artifacts'].items():
        for filename, sha in artifact['files'].items():
            path = directory / 'kernels' / tag / filename
            require(path.resolve().is_relative_to(directory.resolve()), 'Invalid kernel artifact path')
            require(digest(path.read_bytes()) == sha, 'Changed compiled kernel artifact')
    for name, sha in meta['source_files'].items():
        path = directory / 'source' / name
        require(path.resolve().is_relative_to(directory.resolve()), 'Invalid source artifact path')
        require(digest(path.read_bytes()) == sha, 'Changed exported source')


def compare(left, right, output):
    from .cli import load_result, compare_workers, csv_write, seal
    left, right, output = Path(left), Path(right), Path(output)
    require(not output.exists(), 'Comparison exists; choose a new output')
    campaigns = [read_json(p / 'campaign.json') for p in (left, right)]
    require(all(c['status'] == 'complete' for c in campaigns), 'Both intervention campaigns must be complete')
    require(campaigns[0]['protocol'] == campaigns[1]['protocol'] and campaigns[0]['protocol'].get('experiment') == EXPERIMENT,
            'Different intervention protocols or bundles')
    require(campaigns[0]['processes'] == campaigns[1]['processes'], 'Different process counts')
    loaded = []
    restart_ok = []
    for base, campaign in zip((left, right), campaigns):
        workers = []
        require(campaign['processes'] >= 2, 'Need repeated intervention processes')
        for i in range(1, campaign['processes'] + 1):
            directory = base / f'process-{i:03d}'
            meta, arrays = load_result(directory)
            require(meta['protocol'] == campaign['protocol'], 'Worker protocol differs from campaign')
            verify_artifacts(directory, meta)
            require(all(len(r['repeat_sha256']) == campaign['protocol']['repeats'] for r in meta['records'].values()),
                    'Incomplete repetition coverage')
            workers.append((meta, arrays))
        loaded.append(workers)
        restart_ok.append(all(all(r['repeat_bitwise_equal'] for r in m['records'].values()) and
                              set(a) == set(workers[0][1]) and all(np.array_equal(a[k], workers[0][1][k]) for k in a)
                              for m, a in workers))
    effects, coordinates, stages = [], [], []
    for i, ((a, aa), (b, bb)) in enumerate(zip(*loaded), 1):
        report = compare_workers(left / f'process-{i:03d}', right / f'process-{i:03d}')
        stages.extend({'process': i, **r} for r in report['rows'])
        old_a, old_b = aa[key('original', native=True)], bb[key('original', native=True)]
        for variant in PRIMARY:
            k = key(variant, native=True)
            eligible = all(restart_ok) and all(m['controls']['variants'][v]['attribution_eligible']
                                              for m in (a, b) for v in ('original', variant))
            row = {'process': i, 'variant': variant, 'attribution_eligible': eligible,
                   **effect_counts(old_a, old_b, aa[k], bb[k])}
            row['ulp_max'] = compare_values(decode_bits(aa[k], 'bfloat16'), decode_bits(bb[k], 'bfloat16'), 'bfloat16')['ulp_max']
            for label, m in (('left', a), ('right', b)):
                e = m['records'][k]['reference_errors'][-1]
                row[f'{label}_reference_mismatch_percent'] = e['differing_percent']
                row[f'{label}_reference_relative_l2'] = e['relative_l2_error']
            effects.append(row)
            selected = (old_a != old_b) | (aa[k] != bb[k])
            for r, c in np.argwhere(selected):
                coordinates.append({'process': i, 'variant': variant, 'row': int(r), 'column': int(c),
                    'baseline_different': bool(old_a[r,c] != old_b[r,c]), 'now_different': bool(aa[k][r,c] != bb[k][r,c]),
                    'left': float(decode_bits(aa[k], 'bfloat16')[r,c]), 'right': float(decode_bits(bb[k], 'bfloat16')[r,c]),
                    'left_bits': int(aa[k][r,c]), 'right_bits': int(bb[k][r,c]), 'attribution_eligible': eligible})
    output.mkdir(parents=True)
    summary = {'schema_version': 1, 'protocol': campaigns[0]['protocol'], 'left': str(left), 'right': str(right),
               'left_environment': loaded[0][0][0]['environment'], 'right_environment': loaded[1][0][0]['environment'],
               'restart_controls_pass': restart_ok, 'effects': effects,
               'left_controls': [m['controls'] for m, _ in loaded[0]], 'right_controls': [m['controls'] for m, _ in loaded[1]],
               'note': 'Only interpret effects when attribution_eligible is true. Shared outputs test causality; they are not production optimizations or proof of a hardware instruction. Inspect saved assembly and software versions next.'}
    seal(output / 'comparison.json', summary)
    csv_write(output / 'effects.csv', effects)
    csv_write(output / 'coordinates.csv', coordinates)
    csv_write(output / 'stages.csv', stages)
    lines = ['# A100/MI210 normalization interventions', '',
             'Differences count BF16 output elements across the complete captured tensor. Reference error is numerical, not task accuracy.', '',
             '| Process | Variant | Baseline differences | Remaining old | Resolved | New | Total (%) | Controls pass |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in effects:
        lines.append(f"| {r['process']} | {r['variant']} | {r['baseline_differing']} | {r['remaining']} | {r['resolved']} | {r['new']} | {r['differing']} ({r['differing_percent']:.6g}%) | {r['attribution_eligible']} |")
    lines += ['', 'Read [effects.csv](effects.csv), [stages.csv](stages.csv), and [comparison.json](comparison.json). '
              'Nonempty mismatches create `coordinates.csv`, including every new difference.', '',
              'A false control means the intervention is inconclusive: inspect historical baseline reproduction, exact operands, '
              'same-value replay, observer effects, binary identity, and repeats. Even passing controls require inspecting generated '
              'code before attributing the effect to a compiler decision or hardware instruction.', '',
              'The input campaigns contain `kernels/` (PTX/cubin or AMDGCN/HSACO plus IR), `source/`, and `focus-values.csv`. '
              'Keep these folders together when transferring results.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return {'output': str(output), 'effects': effects}


def add_arguments(sub):
    p = sub.add_parser('prepare-intervention', help='Freeze shared FP32 operation values from verified NVIDIA/AMD baseline campaigns')
    p.add_argument('left'); p.add_argument('right'); p.add_argument('--output', required=True)
    p.add_argument('--fixture', default=str(FIXTURE))
    p = sub.add_parser('intervene', help='Run four variants, a replay bridge, and three same-value controls on the captured tensor')
    p.add_argument('--backend', choices=['nvidia', 'amd'], required=True)
    p.add_argument('--bundle', default=str(BUNDLE)); p.add_argument('--output')
    p.add_argument('--repeats', type=int, default=3); p.add_argument('--processes', type=int, default=2)
    p.add_argument('--timeout', type=float, default=900); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('compare-interventions')
    p.add_argument('left'); p.add_argument('right'); p.add_argument('--output', required=True)


def dispatch(args):
    if args.command == 'prepare-intervention':
        return prepare_bundle(args.left, args.right, args.output, args.fixture)
    if args.command == 'intervene':
        return run(args)
    return compare(args.left, args.right, args.output)
