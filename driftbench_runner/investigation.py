"""Portable, controlled A100/MI210 experiments on saved DriftBench cases.

No accelerator package is imported by prepare, report, or the client runner.
Experiments remain normal DriftBench settings, including sandboxed evaluation.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import itertools
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .common import ROOT, WORKLOADS, append_jsonl, digest, keyed, now, read_json, read_jsonl, write_json
from .comparison import verified_evaluations
from .evaluation import run_workloads
from .http_inference import run_http
from .logging import activity, event
from .serving import launch_command, verified_metadata
from .stages import run_stage, verify_inference
from .suite import stop_process

MODELS = ('llama32_1b', 'qwen35_08b')
DEVICES = {'a100': 'nvidia', 'mi210': 'amd'}
CONDITIONS = ('original', 'serial', 'serial-eager', 'serial-eager-fp32')
FIELDS = ('workload', 'prompt_id', 'source_sha256', 'request_sha256', 'prompt', 'rendered_prompt')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def csv_file(path, records):
    records = list(records)
    with Path(path).open('w', newline='') as handle:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def first_difference(a, b):
    """Index at first differing token or termination; None means exact agreement."""
    require(isinstance(a, list) and isinstance(b, list), 'Server output token IDs are required')
    return next((i for i, (x, y) in enumerate(itertools.zip_longest(a, b)) if x != y), None)


def portable_name(name):
    require(bool(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name)), f'Invalid path identifier: {name}')
    return name


def load_bundle(directory):
    root = Path(directory).resolve()
    bundle = read_json(root / 'bundle.json')
    require(bundle.get('schema_version') == 1, 'Unsupported investigation bundle')
    identity = {k: v for k, v in bundle.items() if k != 'fingerprint'}
    require(digest(identity) == bundle['fingerprint'], 'Bundle manifest changed')
    for rel, expected in bundle['files'].items():
        path = (root / rel).resolve()
        require(path.is_relative_to(root) and path.is_file(), f'Invalid bundle file: {rel}')
        require(digest(path.read_bytes()) == expected, f'Bundle content changed: {rel}')
    for slug in bundle['models']:
        portable_name(slug)
    return root, bundle


def stratified_cases(candidates, count):
    """Deterministic round-robin over workload and length-limited status."""
    groups = defaultdict(list)
    for row in candidates:
        groups[(row['workload'], row['length_limited'])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: digest([r['workload'], r['prompt_id']]))
    ordered = []
    for level in itertools.zip_longest(*(groups[k] for k in sorted(groups))):
        ordered.extend(r for r in level if r is not None)
    require(len(ordered) >= count, f'Need {count} cases in each category; only {len(ordered)} available')
    return ordered[:count]


def prepare(inventory_path, output, models=MODELS, per_group=8):
    require(per_group > 0, '--per-group must be positive')
    require(len(set(models)) == len(models), 'Duplicate model selection')
    output = Path(output).resolve()
    require(not output.exists(), 'Bundle output exists; choose a new directory')
    inventory_path = Path(inventory_path).resolve()
    inventory = read_json(inventory_path)
    pairs = [p for p in inventory['comparisons'] if p['candidate_hardware'].lower() == 'mi210']
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.investigation-', dir=output.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        triage, selected_models = [], []
        for pair in pairs:
            slug = portable_name(pair['id'].removesuffix('_a100_vs_mi210'))
            saved = {}
            for device, side in [('a100', 'baseline'), ('mi210', 'candidate')]:
                run = ROOT / pair[side]
                manifest, responses, _ = verify_inference(run)
                scores = verified_evaluations(run, manifest, responses)
                sources = keyed(run_workloads(run, ['code', 'math'])[0])
                saved[device] = (manifest, responses, scores, sources)
            am, a, ae, sources = saved['a100']
            bm, b, be, bsources = saved['mi210']
            for field in ('model', 'revision', 'generation', 'seed', 'prompt_format', 'chat_template_kwargs'):
                require(am['config'].get(field) == bm['config'].get(field), f'Mismatching {field}: {slug}')
            candidates = []
            selected_keys = [key for key in a if key[0] in ('code', 'math')]
            require(set(selected_keys) == {key for key in b if key[0] in ('code', 'math')}, 'Saved code/math selections differ')
            for key in selected_keys:
                source = sources[key]
                ar, br = a[key], b[key]
                require(source == bsources[key], f'Source mismatch: {key}')
                for field in ('input_token_ids', 'rendered_prompt', 'effective_sampling', 'request_sha256'):
                    require(ar.get(field) == br.get(field), f'Input/sampling mismatch: {slug} {key} {field}')
                require(ae[key]['status'] == be[key]['status'] == 'scored', f'Unscored case: {key}')
                ac, bc = ae[key]['correct'], be[key]['correct']
                pos = first_difference(ar['output_token_ids'], br['output_token_ids'])
                category = ('regression' if ac and not bc else 'improvement' if bc and not ac else
                            'control' if pos is None and ar['output_text'] == br['output_text'] else 'changed_same_label')
                require(pos is not None or ac == bc or ar['output_text'] != br['output_text'],
                        f'Identical output has conflicting evaluation: {key}')
                flags = []
                if ar['finish_reason'] == 'length' or br['finish_reason'] == 'length':
                    flags.append('output_limit_review')
                if key[0] == 'code' and any(r['output_text'].count('```') > 2 for r in (ar, br)):
                    flags.append('multiple_code_blocks_review')
                if key[0] == 'math' and any(r['output_text'].count('####') > 1 for r in (ar, br)):
                    flags.append('multiple_final_answers_review')
                item = {'model_key': slug, 'workload': key[0], 'prompt_id': key[1], 'category': category,
                        'a100_correct': ac, 'mi210_correct': bc, 'first_token_difference': pos,
                        'a100_tokens': len(ar['output_token_ids']), 'mi210_tokens': len(br['output_token_ids']),
                        'length_limited': bool('output_limit_review' in flags), 'review_flags': ';'.join(flags),
                        'manual_classification': '', 'review_notes': ''}
                triage.append(item)
                candidates.append(item)
            if slug not in models:
                continue
            selected_models.append(slug)
            chosen = [r for category in ('regression', 'improvement', 'control')
                      for r in stratified_cases([r for r in candidates if r['category'] == category], per_group)]
            # Freeze workload/prompt order for all machines, conditions, repeats.
            chosen.sort(key=lambda r: (r['workload'], r['prompt_id']))
            directory = stage / 'models' / slug
            directory.mkdir(parents=True)
            cases = []
            for item in chosen:
                key = item['workload'], item['prompt_id']
                ar = a[key]
                cases.append({**item, 'source': sources[key],
                              'request': {**{f: ar[f] for f in FIELDS}, 'input_ids': ar['input_token_ids']},
                              'original': {d: {'output_text': saved[d][1][key]['output_text'],
                                               'output_token_ids': saved[d][1][key]['output_token_ids'],
                                               'finish_reason': saved[d][1][key]['finish_reason'],
                                               'evaluation': saved[d][2][key]}
                                           for d in DEVICES}})
            write_json(directory / 'cases.json', cases)
            for device in DEVICES:
                config = deepcopy(saved[device][0]['config'])
                require(config['backend'] == 'vllm' and config['hardware']['vendor'] == DEVICES[device],
                        'Investigation currently supports A100/MI210 vLLM profiles')
                write_json(directory / f'{device}.json', config)
            source_info = {}
            for workload in ('code', 'math'):
                picked = [c['source'] for c in cases if c['workload'] == workload]
                require(picked, 'Selection must cover code and math')
                ids = {s['prompt_id'] for s in picked}
                rest = [s for k, s in sources.items() if k[0] == workload and k[1] not in ids]
                dataset = directory / 'datasets' / WORKLOADS[workload][0]
                dataset.parent.mkdir(exist_ok=True)
                # Full dataset with chosen cases first supports the existing
                # selected-prefix integrity verifier without altering any prompt.
                for source in picked + rest:
                    raw = {k: v for k, v in source.items() if k not in ('workload', 'source_sha256')}
                    require(digest(raw) == source['source_sha256'], 'Source reconstruction changed')
                    append_jsonl(dataset, raw)
                source_info[workload] = {'file': dataset.name, 'sha256': digest(dataset.read_bytes()),
                                         'available': len(picked + rest), 'selected': len(picked)}
            write_json(directory / 'sources.json', source_info)
        require(set(selected_models) == set(models), f'Missing selected models: {set(models) - set(selected_models)}')
        csv_file(stage / 'triage.csv', triage)
        bundle = {'schema_version': 1, 'created_at': now(), 'models': selected_models,
                  'selection': {'per_group': per_group, 'groups': ['regression', 'improvement', 'control'],
                                'method': 'workload/length round-robin, SHA256 order; purposive, not representative'},
                  'original_pairs': pairs,
                  'files': {str(p.relative_to(stage)): digest(p.read_bytes()) for p in sorted(stage.rglob('*')) if p.is_file()}}
        bundle['fingerprint'] = digest(bundle)
        write_json(stage / 'bundle.json', bundle)
        stage.rename(output)
    event('Frozen diagnostic bundle prepared', models=selected_models, cases_per_model=per_group * 3, output=str(output))
    return bundle


def trial_config(template, run_dir, device, condition, port):
    require(condition in CONDITIONS and device in DEVICES, 'Unknown condition/device')
    config = deepcopy(template)
    require(config['backend'] == 'vllm' and config['hardware']['vendor'] == DEVICES[device], 'Wrong device profile')
    config['setup_id'] = f'{device}_{run_dir.parent.parent.name}_{condition}_{run_dir.name}'
    config['server']['base_url'] = f'http://127.0.0.1:{port}'
    config['server']['metadata_path'] = str(run_dir / 'server.json')
    config.setdefault('launch', {})['log_path'] = str(run_dir / 'server.log')
    # Preserve scheduler allocation. Old visibility overrides cannot be portable.
    env = config['launch'].get('env', {})
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        require(key not in env, f'Remove saved {key} override before preparing a portable bundle')
    if condition != 'original':
        config['batch_size'] = 1
        config['batch_size_by_workload'] = {'code': 1, 'math': 1}
        config['engine']['max_num_seqs'] = 1
    if condition in ('serial-eager', 'serial-eager-fp32'):
        config['engine']['enforce_eager'] = True
    if condition == 'serial-eager-fp32':
        config['engine']['dtype'] = 'float32'
    launch_command(config)
    return config


@contextmanager
def owned_server(config_path, server_python):
    config = read_json(config_path)
    directory = Path(config_path).parent
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1',
                   'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    old = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        with (directory / 'launcher.log').open('a') as log:
            process = subprocess.Popen([str(server_python), '-m', 'driftbench_runner', 'serve', '--config', str(config_path)],
                                       cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + config['launch'].get('startup_timeout_seconds', 900) + config['server'].get('timeout_seconds', 900)
                with activity('Starting fresh investigation server', stage='startup', log=str(directory / 'launcher.log')):
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f'Server exited {process.returncode}; see {directory / "launcher.log"}')
                        metadata = Path(config['server']['metadata_path'])
                        if metadata.exists():
                            status = read_json(metadata)
                            if status.get('launcher_pid') == process.pid and status.get('status') == 'ready':
                                verified_metadata(config)
                                break
                            if status.get('status') == 'failed':
                                raise RuntimeError(status.get('error', 'Server failed'))
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f'Server startup timeout; see {directory / "launcher.log"}')
                        time.sleep(1)
                yield
            finally:
                stop_process(process, launcher=True)
                metadata = Path(config['server']['metadata_path'])
                if metadata.exists():
                    final = read_json(metadata)
                    require(final.get('launcher_pid') != process.pid or 'stopped_at' in final,
                            f'Server cleanup unconfirmed; inspect {directory / "launcher.log"} before continuing')
    finally:
        signal.signal(signal.SIGTERM, old)


def response_identity(outputs):
    return digest([[*key, digest(row)] for key, row in sorted(outputs.items())])


def check_trial(path, fingerprint=None):
    info = read_json(path / 'experiment.json')
    require(info.get('status') == 'complete', f'Incomplete trial: {path}; use a new output root to retry it')
    if fingerprint:
        require(info['bundle_fingerprint'] == fingerprint, f'Different case bundle: {path}')
    manifest, outputs, _ = verify_inference(path)
    require(info['response_sha256'] == response_identity(outputs), f'Changed response records: {path}')
    require(info['run_fingerprint'] == manifest['run_fingerprint'], f'Changed trial identity: {path}')
    require(info['config_sha256'] == digest(manifest['config']), f'Changed trial config: {path}')
    require(manifest['config']['hardware']['vendor'] == DEVICES.get(info['device']), f'Trial device metadata differs: {path}')
    require(info['condition'] in CONDITIONS and type(info['repeat']) is int and info['repeat'] > 0, 'Invalid trial condition/repeat')
    return info, manifest, outputs


def run_trials(bundle_dir, device, output, server_python, models=None, conditions=('original', 'serial'),
               repeats=3, port=8001, resume=False, dry_run=False):
    root, bundle = load_bundle(bundle_dir)
    models = models or bundle['models']
    require(set(models) <= set(bundle['models']) and len(set(models)) == len(models), 'Invalid model selection')
    require(repeats > 0 and 0 < port < 65536, 'Positive repeats and valid port required')
    require(conditions and len(set(conditions)) == len(conditions), 'Duplicate/empty conditions')
    output = Path(output).resolve()
    # Do not resolve the Python symlink; doing so escapes its virtualenv.
    server_python = Path(server_python).expanduser().absolute()
    if not dry_run:
        require(server_python.is_file() and os.access(server_python, os.X_OK), f'Missing serving Python: {server_python}')
    plan = []
    for slug, condition, repeat in itertools.product(models, conditions, range(1, repeats + 1)):
        run_dir = output / device / slug / condition / f'repeat-{repeat:03d}'
        config = trial_config(read_json(root / 'models' / slug / f'{device}.json'), run_dir, device, condition, port)
        plan.append({'model_key': slug, 'condition': condition, 'repeat': repeat, 'path': str(run_dir), 'config': config})
    if dry_run:
        return {'bundle_fingerprint': bundle['fingerprint'], 'trials': plan, 'server_python': str(server_python)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'.{device}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in plan:
            run_dir, config = Path(job['path']), job['config']
            if run_dir.exists():
                require(resume, f'Trial exists: {run_dir}; --resume only skips complete identical trials')
                _, meta, _ = check_trial(run_dir, bundle['fingerprint'])
                require(meta['config'] == config, 'Resume configuration changed; choose a new output root')
                continue
            run_dir.mkdir(parents=True)
            source_dir = root / 'models' / job['model_key']
            shutil.copytree(source_dir / 'datasets', run_dir / 'datasets')
            cases = read_json(source_dir / 'cases.json')
            rows = [c['request'] for c in cases]
            sources = read_json(source_dir / 'sources.json')
            config_path = run_dir / 'config.json'
            write_json(config_path, config)
            info = {'schema_version': 1, 'bundle_fingerprint': bundle['fingerprint'], 'device': device,
                    'model_key': job['model_key'], 'condition': job['condition'], 'repeat': job['repeat'],
                    'config_sha256': digest(config), 'created_at': now(), 'status': 'running',
                    'server_python': str(server_python), 'fresh_server': True}
            write_json(run_dir / 'experiment.json', info)
            try:
                with owned_server(config_path, server_python):
                    with (run_dir / '.run.lock').open('a') as run_lock:
                        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        run_http(config, rows, sources, run_dir)
                manifest, outputs, _ = verify_inference(run_dir)
                require(all(r.get('output_token_ids') is not None for r in outputs.values()), 'Missing server token IDs')
                info.update(status='complete', completed_at=now(), response_sha256=response_identity(outputs),
                            run_fingerprint=manifest['run_fingerprint'])
                write_json(run_dir / 'experiment.json', info)
            except BaseException as exc:
                info.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                write_json(run_dir / 'experiment.json', info)
                raise
    return {'status': 'complete', 'trials': len(plan), 'output': str(output)}


def discover_trials(roots, fingerprint=None):
    trials, seen = [], set()
    paths = set()
    for root in roots:
        root = Path(root).resolve()
        paths.update([root / 'experiment.json'] if (root / 'experiment.json').exists() else root.rglob('experiment.json'))
    for path in sorted(paths):
        info, manifest, outputs = check_trial(path.parent, fingerprint)
        key = tuple(info[k] for k in ('device', 'model_key', 'condition', 'repeat'))
        require(key not in seen, f'Duplicate trial identity: {key}')
        seen.add(key)
        trials.append((path.parent, info, manifest, outputs))
    require(trials, 'No complete investigation trials found')
    return trials


def evaluate(roots):
    trials = discover_trials(roots)
    for path, _, _, _ in trials:
        run_stage(path, 'code')
        run_stage(path, 'final')
    return {'evaluated_trials': len(trials)}


def report(bundle_dir, roots, output):
    root, bundle = load_bundle(bundle_dir)
    trials = discover_trials(roots, bundle['fingerprint'])
    output = Path(output).resolve()
    require(not output.exists(), 'Report output exists; choose a new directory')
    evaluations, accuracies, rows, summaries, probe_cases = {}, [], [], [], []
    evaluator_protocols = defaultdict(set)
    for path, info, meta, outputs in trials:
        from urllib.parse import urlsplit
        template = read_json(root / 'models' / info['model_key'] / f"{info['device']}.json")
        expected_config = trial_config(template, Path(meta['config']['server']['metadata_path']).parent,
                                       info['device'], info['condition'], urlsplit(meta['config']['server']['base_url']).port)
        require(meta['config'] == expected_config, f'Trial deviates from its declared condition: {path}')
        cases = read_json(root / 'models' / info['model_key'] / 'cases.json')
        expected = keyed([c['request'] for c in cases])
        require(set(outputs) == set(expected), f'Trial has a different case selection: {path}')
        for key, record in outputs.items():
            require(record['input_token_ids'] == expected[key]['input_ids'] and
                    record['request_sha256'] == expected[key]['request_sha256'], 'Trial input changed')
        scores = verified_evaluations(path, meta, outputs) if (path / 'evaluation.json').exists() else {}
        evaluations[str(path)] = scores
        for workload in ('code', 'math'):
            group = [r for k, r in scores.items() if k[0] == workload and r['status'] == 'scored']
            expected_count = sum(k[0] == workload for k in outputs)
            for score in group:
                evaluator_protocols[workload].add(digest({k: score.get(k) for k in ('evaluator_version', 'method', 'execution_environment')}))
            accuracies.append({**{k: info[k] for k in ('device', 'model_key', 'condition', 'repeat')},
                               'workload': workload, 'expected': expected_count, 'scored': len(group),
                               'correct': sum(r['correct'] for r in group),
                               'accuracy_percent': 100 * sum(r['correct'] for r in group) / len(group) if len(group) == expected_count and group else None,
                               'evaluation_complete': len(group) == expected_count})
    require(all(len(p) <= 1 for p in evaluator_protocols.values()), 'Evaluator protocols/environments differ; score all trials on one evaluation host')
    for left, right in itertools.combinations(trials, 2):
        lp, li, lm, lo = left
        rp, ri, rm, ro = right
        if li['model_key'] != ri['model_key']:
            continue
        within = li['device'] == ri['device'] and li['condition'] == ri['condition']
        cross = li['device'] != ri['device'] and li['condition'] == ri['condition']
        ablation = (li['device'] == ri['device'] and li['repeat'] == ri['repeat'] and
                    abs(CONDITIONS.index(li['condition']) - CONDITIONS.index(ri['condition'])) == 1)
        if not (within or cross or ablation):
            continue
        if cross and li['device'] != 'a100':
            lp, li, lm, lo, rp, ri, rm, ro = rp, ri, rm, ro, lp, li, lm, lo
        if ablation and CONDITIONS.index(li['condition']) > CONDITIONS.index(ri['condition']):
            lp, li, lm, lo, rp, ri, rm, ro = rp, ri, rm, ro, lp, li, lm, lo
        kind = 'within_device' if within else 'cross_device' if cross else 'condition_change'
        require(set(lo) == set(ro), 'Unpaired trial outputs')
        pair_rows = []
        for key in lo:
            a, b = lo[key], ro[key]
            pos = first_difference(a['output_token_ids'], b['output_token_ids'])
            sa, sb = evaluations[str(lp)].get(key), evaluations[str(rp)].get(key)
            scored = sa and sb and sa['status'] == sb['status'] == 'scored'
            record = {'kind': kind, 'model_key': li['model_key'], 'left_device': li['device'], 'right_device': ri['device'],
                      'left_condition': li['condition'], 'right_condition': ri['condition'],
                      'left_repeat': li['repeat'], 'right_repeat': ri['repeat'], 'workload': key[0], 'prompt_id': key[1],
                      'token_equal': pos is None, 'first_token_difference': pos,
                      'left_token': a['output_token_ids'][pos] if pos is not None and pos < len(a['output_token_ids']) else None,
                      'right_token': b['output_token_ids'][pos] if pos is not None and pos < len(b['output_token_ids']) else None,
                      'text_equal': a['output_text'] == b['output_text'], 'left_finish': a['finish_reason'], 'right_finish': b['finish_reason'],
                      'left_correct': sa['correct'] if scored else None, 'right_correct': sb['correct'] if scored else None,
                      'label_flip': sa['correct'] != sb['correct'] if scored else None,
                      'left_run': str(lp), 'right_run': str(rp)}
            pair_rows.append(record)
            if cross and pos is not None and li['repeat'] == ri['repeat'] == 1:
                prefix = a['input_token_ids'] + a['output_token_ids'][:pos]
                probe_cases.append({'model_key': li['model_key'], 'condition': li['condition'],
                                    'workload': key[0], 'prompt_id': key[1], 'position': pos,
                                    'input_ids': prefix, 'prefix_sha256': digest(prefix),
                                    'left_token': record['left_token'], 'right_token': record['right_token']})
        rows.extend(pair_rows)
        for workload in ('code', 'math'):
            group = [r for r in pair_rows if r['workload'] == workload]
            scored = [r for r in group if r['label_flip'] is not None]
            summaries.append({**{k: group[0][k] for k in ('kind', 'model_key', 'left_device', 'right_device', 'left_condition', 'right_condition', 'left_repeat', 'right_repeat')},
                              'workload': workload, 'paired': len(group),
                              'token_agreement_percent': 100 * sum(r['token_equal'] for r in group) / len(group),
                              'scored_pairs': len(scored), 'flip_percent': 100 * sum(r['label_flip'] for r in scored) / len(scored) if len(scored) == len(group) and scored else None,
                              'regressions': sum(r['left_correct'] and not r['right_correct'] for r in scored),
                              'improvements': sum(not r['left_correct'] and r['right_correct'] for r in scored),
                              'environment_differs': lm['environment'] != rm['environment']})
    output.mkdir(parents=True)
    csv_file(output / 'agreement.csv', summaries)
    csv_file(output / 'accuracy.csv', accuracies)
    csv_file(output / 'first-divergences.csv', rows)
    probe = {'schema_version': 1, 'bundle_fingerprint': bundle['fingerprint'], 'cases': probe_cases}
    probe['fingerprint'] = digest(probe)
    write_json(output / 'probe-cases.json', probe)
    result = {'bundle_fingerprint': bundle['fingerprint'], 'trials': len(trials),
              'trial_coverage': [{k: info[k] for k in ('model_key', 'device', 'condition', 'repeat')} for _, info, _, _ in trials],
              'evaluation_complete': all(r['evaluation_complete'] for r in accuracies),
              'agreements': summaries, 'accuracy': accuracies,
              'limitations': ['Purposively selected cases; not full-benchmark accuracy.',
                             'Repeated pair comparisons are dependent observations, not independent statistical samples.',
                             'Cross-device trials include software/backend differences.',
                             'Logits and intermediate tensors are not captured by the normal inference path.']}
    write_json(output / 'report.json', result)
    (output / 'README.md').write_text(
        '**A100–MI210 investigation**\n\n'
        f"{len(trials)} completed trials. Evaluation complete: {result['evaluation_complete']}.\n\n"
        '[Accuracy](accuracy.csv) · [Agreement and flip rates](agreement.csv) · '
        '[First differing tokens](first-divergences.csv) · [Same-prefix probe requests](probe-cases.json)\n\n'
        'Inspect within-device variation before attributing cross-device differences. '
        'Condition-change rows compare controlled interventions; cross-device rows include all repeat combinations. '
        'Trial coverage is explicit in report.json; absent models/conditions are not treated as passing.\n\n'
        'Accuracy is only for the frozen diagnostic subset. Missing evaluations remain null. '
        'Bitwise tensor and raw-logit results require separate captures; top-k probes report log-probabilities, not raw logits.\n')
    return {'trials': len(trials), 'pair_comparisons': len(summaries), 'evaluation_complete': result['evaluation_complete'], 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Verify saved runs and freeze portable diagnostic cases')
    p.add_argument('--inventory', default=str(ROOT / 'results/comparisons/a100-baseline-20260916/inventory.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--models', nargs='+', default=list(MODELS))
    p.add_argument('--per-group', type=int, default=8)
    p = sub.add_parser('run', help='Launch fresh servers and generate repeated trials')
    p.add_argument('--bundle', required=True)
    p.add_argument('--device', choices=list(DEVICES), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--server-python', required=True)
    p.add_argument('--models', nargs='+')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=['original', 'serial'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('evaluate', help='Evaluate copied trials on the common code-evaluation host')
    p.add_argument('roots', nargs='+')
    p = sub.add_parser('report', help='Summarize token/accuracy agreement and make same-prefix probe requests')
    p.add_argument('--bundle', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    from .investigation_probes import add_parsers, dispatch
    add_parsers(sub)
    from . import causal
    causal.add_parsers(sub)
    args = parser.parse_args(argv)
    from .credentials import load_credentials
    from .common import local_environment
    local_environment()
    if args.command in ('run', 'probe', 'capture-reference'):
        load_credentials()
    if args.command == 'prepare':
        result = prepare(args.inventory, args.output, args.models, args.per_group)
        result = {k: result[k] for k in ('models', 'fingerprint', 'selection')}
    elif args.command == 'run':
        result = run_trials(args.bundle, args.device, args.output, args.server_python, args.models,
                            args.conditions, args.repeats, args.port, args.resume, args.dry_run)
    elif args.command == 'evaluate':
        result = evaluate(args.roots)
    elif args.command == 'report':
        result = report(args.bundle, args.runs, args.output)
    elif args.command in causal.COMMANDS:
        result = causal.dispatch(args)
    else:
        result = dispatch(args)
    import json
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
