"""Optional same-prefix log-probability probes and primitive precision diagnostics."""
from copy import deepcopy
import math
from pathlib import Path
import re

from .common import digest, now, read_json, write_json
from .http_backend import HTTPBackend


class TopKBackend(HTTPBackend):
    def __init__(self, config, top_k):
        super().__init__(config)
        self.top_k = top_k

    def payload(self, row, max_tokens=None):
        path, payload = super().payload(row, max_tokens)
        payload.update(logprobs=self.top_k, return_tokens_as_token_ids=True)
        return path, payload


def parse_topk(response, k):
    from .investigation import require
    values = response['server_response']['choices'][0].get('logprobs')
    require(values and len(values.get('top_logprobs') or []) == 1, 'Endpoint did not return one-step top-k log-probabilities')
    tokens = []
    for token, score in values['top_logprobs'][0].items():
        require(re.fullmatch(r'token_id:\d+', token) is not None, 'Endpoint did not identify log-probability candidates by token ID')
        require(isinstance(score, (int, float)) and math.isfinite(score), 'Invalid log-probability')
        tokens.append({'token_id': int(token.split(':')[1]), 'logprob': score})
    tokens.sort(key=lambda r: (-r['logprob'], r['token_id']))
    require(len(tokens) >= k, f'Endpoint returned fewer than {k} top candidates')
    return tokens[:k]


def probe(bundle_dir, requests_path, device, model, condition, output, server_python, top_k=10, port=8001):
    from .investigation import load_bundle, require, trial_config, owned_server
    from .serving import verified_metadata
    root, bundle = load_bundle(bundle_dir)
    requests = read_json(requests_path)
    require(requests['bundle_fingerprint'] == bundle['fingerprint'], 'Probe requests belong to a different bundle')
    require(requests['fingerprint'] == digest({k: v for k, v in requests.items() if k != 'fingerprint'}), 'Probe requests changed')
    require(model in bundle['models'] and 2 <= top_k <= 20, 'Choose a bundled model and top-k between 2 and 20')
    cases = [r for r in requests['cases'] if r['model_key'] == model and r['condition'] == condition]
    require(cases, 'No matching differing-token cases; generate a report with paired device trials first')
    output = Path(output).resolve()
    require(not output.exists(), 'Probe output exists; choose a new directory')
    require(Path(server_python).is_file(), f'Missing serving Python: {server_python}')
    source_cases = {(c['workload'], c['prompt_id']): c for c in read_json(root / 'models' / model / 'cases.json')}
    for case in cases:
        original = source_cases[(case['workload'], case['prompt_id'])]['request']['input_ids']
        require(case['input_ids'][:len(original)] == original and len(case['input_ids']) == len(original) + case['position'],
                'Probe prefix is inconsistent with the frozen input')
        require(case['prefix_sha256'] == digest(case['input_ids']), 'Probe prefix changed')
    output.mkdir(parents=True)
    config = trial_config(read_json(root / 'models' / model / f'{device}.json'), output, device, condition, port)
    require(all(len(c['input_ids']) + 1 <= config['engine']['max_model_len'] for c in cases), 'Probe exceeds context limit')
    write_json(output / 'config.json', config)
    result = {'schema_version': 1, 'bundle_fingerprint': bundle['fingerprint'], 'requests_fingerprint': requests['fingerprint'],
              'model_key': model, 'device': device, 'condition': condition, 'top_k': top_k, 'records': [],
              'status': 'running', 'score_type': 'API log-probabilities, not raw logits',
              'execution': 'one request at a time, full prefix prefilled on a fresh server; not the original decode/cache state'}
    write_json(output / 'probe.json', result)
    try:
        with owned_server(output / 'config.json', Path(server_python).absolute()):
            result['server_provenance'] = deepcopy(verified_metadata(config))
            backend = TopKBackend(config, top_k)
            for case in cases:
                response = backend.generate_one(case, max_tokens=1)
                top = parse_topk(response, top_k)
                result['records'].append({**case, 'top_k': top, 'top1_top2_logprob_margin': top[0]['logprob'] - top[1]['logprob'],
                                          'response': response})
                write_json(output / 'probe.json', result)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'probe.json', result)
    return {'status': result['status'], 'probed': len(cases), 'output': str(output)}


def compare_probes(left, right, output):
    from .investigation import require, csv_file
    a, b = read_json(left), read_json(right)
    require(a['status'] == b['status'] == 'complete', 'Finish both probe runs first')
    for key in ('bundle_fingerprint', 'requests_fingerprint', 'model_key', 'condition', 'top_k', 'score_type'):
        require(a[key] == b[key], f'Probe mismatch: {key}')
    key = lambda r: (r['workload'], r['prompt_id'], r['position'])
    aa, bb = {key(r): r for r in a['records']}, {key(r): r for r in b['records']}
    require(len(aa) == len(a['records']) and len(bb) == len(b['records']) and aa.keys() == bb.keys(), 'Unpaired or duplicate probe cases')
    rows = []
    for k in aa:
        x, y = aa[k], bb[k]
        require(x['input_ids'] == y['input_ids'] and digest(x['input_ids']) == x['prefix_sha256'] == y['prefix_sha256'], 'Probe histories differ')
        ta = {r['token_id']: r['logprob'] for r in x['top_k']}
        tb = {r['token_id']: r['logprob'] for r in y['top_k']}
        common = ta.keys() & tb.keys()
        rows.append({'workload': k[0], 'prompt_id': k[1], 'position': k[2], 'prefix_sha256': x['prefix_sha256'],
                     'left_device': a['device'], 'right_device': b['device'],
                     'top1_equal': x['top_k'][0]['token_id'] == y['top_k'][0]['token_id'],
                     'top_k_overlap_percent': 100 * len(common) / a['top_k'],
                     'max_common_logprob_error': max((abs(ta[t] - tb[t]) for t in common), default=None),
                     'left_margin': x['top1_top2_logprob_margin'], 'right_margin': y['top1_top2_logprob_margin'],
                     'score_type': a['score_type']})
    require(not Path(output).exists(), 'Output exists')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    csv_file(output, rows)
    return {'paired_probes': len(rows), 'output': str(output)}


def tensor_metrics(a, b):
    """Accept NumPy arrays with the same logical layout and representation."""
    import numpy as np
    from .investigation import require
    require(a.shape == b.shape and a.dtype == b.dtype and a.dtype.kind in 'fiu', 'Tensor shape/dtype must match and be numeric')
    require(a.size > 0, 'Empty tensors cannot be compared')
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    equal_bytes = (a.view(np.uint8).reshape(-1, a.dtype.itemsize) == b.view(np.uint8).reshape(-1, b.dtype.itemsize)).all(axis=1)
    bits = {'shape': list(a.shape), 'dtype': str(a.dtype), 'bitwise_equal': bool(equal_bytes.all()),
            'bitwise_equal_elements_percent': float(equal_bytes.mean() * 100)}
    if a.dtype.kind in 'iu':
        return {**bits, 'numeric_error_note': 'Integer storage compared bitwise only; no floating-point interpretation is assumed.'}
    x, y = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    err = np.abs(x[finite] - y[finite])
    return {**bits, 'nonfinite_pairs': int((~finite).sum()), 'max_absolute_error': float(err.max()) if err.size else None,
            'mean_absolute_error': float(err.mean()) if err.size else None,
            'relative_l2_error': float(np.linalg.norm(err) / max(float(np.linalg.norm(x[finite])), 1e-30)) if err.size else None}


def compare_tensors(left, right, output):
    import numpy as np
    from .investigation import require
    require(not Path(output).exists(), 'Output exists')
    # No pickle: captures contain arrays only, not executable Python objects.
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), 'Tensor capture keys differ')
        records = {k: tensor_metrics(a[k], b[k]) for k in a.files}
    result = {'left_sha256': digest(Path(left).read_bytes()), 'right_sha256': digest(Path(right).read_bytes()),
              'arrays': records, 'note': 'Align tensor semantics and element order before comparison; integer bit views are not floating-point values.'}
    write_json(output, result)
    return result


def capture_reference(bundle_dir, requests_path, model_key, condition, output, device='cuda', dtype='bfloat16', layers=(0, -1), limit=3):
    """Instrument a separate Transformers forward pass, never claim vLLM capture."""
    import numpy as np
    import torch
    import transformers
    from .investigation import load_bundle, require
    root, bundle = load_bundle(bundle_dir)
    requests = read_json(requests_path)
    require(requests['fingerprint'] == digest({k: v for k, v in requests.items() if k != 'fingerprint'}), 'Probe requests changed')
    require(requests['bundle_fingerprint'] == bundle['fingerprint'], 'Different bundle')
    require(model_key in bundle['models'] and limit > 0, 'Choose a bundled model and positive limit')
    cases = [r for r in requests['cases'] if r['model_key'] == model_key and r['condition'] == condition][:limit]
    require(cases, 'No matching probe prefixes')
    require(device == 'cpu' or torch.cuda.is_available(), 'No usable CUDA/HIP accelerator')
    output = Path(output).resolve()
    require(not output.exists(), 'Capture output exists')
    config = read_json(root / 'models' / model_key / 'a100.json')
    # Do not download arbitrary remote code or use a moving model revision.
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config['model'], revision=config['revision'], torch_dtype=getattr(torch, dtype),
        attn_implementation='eager', trust_remote_code=False).to(device).eval()
    original = {(c['workload'], c['prompt_id']): c for c in read_json(root / 'models' / model_key / 'cases.json')}
    arrays, records = {}, []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            base = original[(case['workload'], case['prompt_id'])]['request']['input_ids']
            require(case['input_ids'][:len(base)] == base and digest(case['input_ids']) == case['prefix_sha256'], 'Invalid capture input')
            ids = torch.tensor([case['input_ids']], dtype=torch.long, device=device)
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
            require(result.hidden_states is not None, 'Model does not expose hidden states')
            names = {}
            for layer in layers:
                index = layer if layer >= 0 else len(result.hidden_states) + layer
                require(0 <= index < len(result.hidden_states), f'Invalid hidden-state index {layer}')
                names[f'case_{i}_hidden_{index}'] = result.hidden_states[index][0]
            names[f'case_{i}_logits'] = result.logits[0, -1]
            representations = {}
            for name, tensor in names.items():
                tensor = tensor.detach().contiguous().cpu()
                require(tensor.dtype in (torch.float32, torch.bfloat16), 'Unsupported capture representation')
                arrays[name] = tensor.float().numpy()
                arrays[name + '__bits'] = tensor.view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32).numpy()
                representations[name] = str(tensor.dtype)
            records.append({'workload': case['workload'], 'prompt_id': case['prompt_id'], 'position': case['position'],
                            'prefix_sha256': case['prefix_sha256'], 'input_ids': case['input_ids'], 'representations': representations})
    output.mkdir(parents=True)
    np.savez(output / 'tensors.npz', **arrays)
    metadata = {'schema_version': 1, 'model': config['model'], 'revision': config['revision'],
                'bundle_fingerprint': bundle['fingerprint'], 'requests_fingerprint': requests['fingerprint'],
                'model_key': model_key, 'condition': condition, 'dtype': dtype, 'layers': list(layers),
                'execution': 'Transformers eager full-prefix reference; not the serving vLLM kernel', 'tf32': False,
                'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
                'hip': torch.version.hip, 'device': torch.cuda.get_device_name() if device != 'cpu' else 'CPU',
                'records': records, 'tensors_sha256': digest((output / 'tensors.npz').read_bytes())}
    write_json(output / 'capture.json', metadata)
    return {'captured_prefixes': len(records), 'output': str(output), 'execution': metadata['execution']}


def compare_captures(left, right, output):
    from .investigation import require
    a, b = read_json(Path(left) / 'capture.json'), read_json(Path(right) / 'capture.json')
    for field in ('model', 'revision', 'bundle_fingerprint', 'requests_fingerprint', 'dtype', 'layers', 'execution', 'records'):
        require(a[field] == b[field], f'Incomparable reference captures: {field}')
    for root, metadata in ((left, a), (right, b)):
        require(metadata['tensors_sha256'] == digest((Path(root) / 'tensors.npz').read_bytes()), 'Capture tensors changed')
    result = compare_tensors(Path(left) / 'tensors.npz', Path(right) / 'tensors.npz', output)
    result.update(execution=a['execution'], left_environment={k: a[k] for k in ('torch', 'transformers', 'device', 'cuda', 'hip')},
                  right_environment={k: b[k] for k in ('torch', 'transformers', 'device', 'cuda', 'hip')})
    write_json(output, result)
    return result


def microbench(device, output):
    """Small fixed-input diagnostics, not a tensor-core utilization claim."""
    import numpy as np
    import torch
    from .investigation import require
    output = Path(output).resolve()
    require(not output.exists(), 'Microbenchmark output exists')
    require(device == 'cpu' or torch.cuda.is_available(), 'No usable CUDA/HIP device in this interpreter')
    output.mkdir(parents=True)
    # Disable optional lower-precision FP32 matmul on NVIDIA for this reference.
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = False
    boundary = []
    for scale in (0.5, 1., 2., 4.):
        # Three FP32 points around the BF16 midpoint above the power of two.
        middle = np.float32(scale * (1 + 1 / 256))
        for sign in (1., -1.):
            boundary.extend(sign * float(x) for x in (np.nextafter(middle, np.float32(-np.inf)), middle, np.nextafter(middle, np.float32(np.inf))))
    boundary += [0., -0., 2. ** -133, -(2. ** -133), 2. ** -126]
    arrays, records = {}, []

    def save(name, actual, reference, inputs):
        actual = actual.detach().contiguous().cpu()
        values = actual.float().numpy()
        reference = np.asarray(reference, dtype=np.float64)
        arrays[name] = values
        arrays[name + '__bits'] = actual.view(torch.int16).numpy() if actual.dtype == torch.bfloat16 else actual.view(torch.int32).numpy()
        err = np.abs(values.astype(np.float64) - reference)
        records.append({'operation': name, 'dtype': str(actual.dtype), 'shape': list(actual.shape),
                        'inputs_sha256': digest(inputs), 'max_absolute_error_vs_fp64': float(err.max()),
                        'mean_absolute_error_vs_fp64': float(err.mean()),
                        'relative_l2_error_vs_fp64': float(np.linalg.norm(err) / max(float(np.linalg.norm(reference)), 1e-30))})

    with torch.inference_mode():
        x = torch.tensor(boundary, dtype=torch.float32, device=device)
        save('bf16_conversion', x.bfloat16(), boundary, boundary)
        # These controls may be exact/optimized; not evidence of rounding direction.
        bx = x.bfloat16()
        quantized = bx.float().cpu().numpy().astype(np.float64)
        save('zero_plus_x', torch.zeros_like(bx) + bx, quantized, boundary)
        save('x_plus_zero', bx + torch.zeros_like(bx), quantized, boundary)
        other = torch.full_like(bx, 2. ** -8)
        save('bf16_add', bx + other, quantized + 2. ** -8, boundary)
        save('bf16_multiply', bx * torch.tensor(1.0078125, dtype=torch.bfloat16, device=device), quantized * 1.0078125, boundary)
        for exponent in (-4, 0, 4):
            vals = [v * 2. ** exponent for v in ([256., 1., -256., 1.] * 32)]
            terms = torch.tensor(vals, dtype=torch.bfloat16, device=device)
            ref = np.float64(sum(vals))
            low = torch.zeros((), dtype=torch.bfloat16, device=device)
            high = torch.zeros((), dtype=torch.float32, device=device)
            for term in terms:
                low = low + term
                high = high + term.float()
            save(f'reduction_low_{exponent}', low, ref, vals)
            save(f'reduction_fp32_{exponent}', high, ref, vals)
            save(f'reduction_late_cast_{exponent}', high.bfloat16(), ref, vals)
        # Integer arithmetic makes host inputs independent of RNG/library version.
        a = ((np.arange(32 * 64).reshape(32, 64) * 17) % 67 - 33).astype(np.float32) / 32
        b = ((np.arange(64 * 16).reshape(64, 16) * 13) % 59 - 29).astype(np.float32) / 32
        ta = torch.tensor(a, device=device).bfloat16()
        tb = torch.tensor(b, device=device).bfloat16()
        reference = a.astype(np.float64) @ b.astype(np.float64)
        inputs = [a.tolist(), b.tolist()]
        save('matmul_bf16', ta @ tb, reference, inputs)
        save('matmul_fp32', ta.float() @ tb.float(), reference, inputs)
    np.savez(output / 'tensors.npz', **arrays)
    result = {'schema_version': 1, 'created_at': now(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'hip': torch.version.hip, 'device': device,
              'device_name': torch.cuda.get_device_name() if device != 'cpu' else 'CPU',
              'reference': 'CPU NumPy float64; fixed representable inputs for arithmetic, original FP32 inputs for conversion',
              'tf32_allowed': False, 'operations': records,
              'implementation_sha256': digest(Path(__file__).read_bytes()),
              'tensors_sha256': digest((output / 'tensors.npz').read_bytes()),
              'limits': 'Eager PyTorch operations; no claim of a specific instruction or tensor-core path. CPU checks validate the harness, not A100/MI210 behavior. __bits arrays are raw storage, not numeric error measurements.'}
    write_json(output / 'microbench.json', result)
    return result


def compare_microbench(left, right, output):
    from .investigation import require
    a, b = read_json(Path(left) / 'microbench.json'), read_json(Path(right) / 'microbench.json')
    for field in ('schema_version', 'implementation_sha256', 'reference', 'tf32_allowed'):
        require(a[field] == b[field], f'Microbench implementations/settings differ: {field}')
    control = lambda m: [{k: r[k] for k in ('operation', 'dtype', 'shape', 'inputs_sha256')} for r in m['operations']]
    require(control(a) == control(b), 'Microbench inputs or representations differ')
    for root, metadata in ((left, a), (right, b)):
        require(digest((Path(root) / 'tensors.npz').read_bytes()) == metadata['tensors_sha256'], 'Microbench arrays changed')
    result = compare_tensors(Path(left) / 'tensors.npz', Path(right) / 'tensors.npz', output)
    result.update(left_environment={k: a[k] for k in ('torch', 'cuda', 'hip', 'device_name')},
                  right_environment={k: b[k] for k in ('torch', 'cuda', 'hip', 'device_name')},
                  reference=a['reference'], implementation_sha256=a['implementation_sha256'])
    write_json(output, result)
    return result


def add_parsers(sub):
    p = sub.add_parser('probe', help='Fresh-server, same-prefix next-token top-k log-probability probes')
    for name in ('bundle', 'requests', 'device', 'model', 'condition', 'output', 'server-python'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--port', type=int, default=8001)
    p = sub.add_parser('compare-probes')
    p.add_argument('left')
    p.add_argument('right')
    p.add_argument('--output', required=True)
    p = sub.add_parser('microbench', help='Run with a torch-enabled CUDA/HIP interpreter, or CPU for harness testing')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare-tensors', help='Compare matching numeric arrays in NPZ captures (no pickle)')
    p.add_argument('left')
    p.add_argument('right')
    p.add_argument('--output', required=True)
    p = sub.add_parser('capture-reference', help='Optional Transformers forward-pass logits/hidden states; separate from vLLM')
    for name in ('bundle', 'requests', 'model', 'condition', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--layers', type=int, nargs='+', default=[0, -1])
    p.add_argument('--limit', type=int, default=3)
    p = sub.add_parser('compare-captures', help='Verify capture provenance then compare bits, tensors, and raw logits')
    p.add_argument('left')
    p.add_argument('right')
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare-microbench', help='Verify matching synthetic inputs/implementation before comparing arrays')
    p.add_argument('left')
    p.add_argument('right')
    p.add_argument('--output', required=True)


def dispatch(args):
    if args.command == 'probe':
        return probe(args.bundle, args.requests, args.device, args.model, args.condition, args.output,
                     args.server_python, args.top_k, args.port)
    if args.command == 'compare-probes':
        return compare_probes(args.left, args.right, args.output)
    if args.command == 'microbench':
        return microbench(args.device, args.output)
    if args.command == 'capture-reference':
        return capture_reference(args.bundle, args.requests, args.model, args.condition, args.output,
                                 args.device, args.dtype, args.layers, args.limit)
    if args.command == 'compare-captures':
        return compare_captures(args.left, args.right, args.output)
    if args.command == 'compare-microbench':
        return compare_microbench(args.left, args.right, args.output)
    return compare_tensors(args.left, args.right, args.output)
