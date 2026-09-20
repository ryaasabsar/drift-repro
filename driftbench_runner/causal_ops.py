"""Portable tensor artifacts and explicit reference implementations for replays.

NPZ files contain numeric arrays only; loading never unpickles executable code.
References specify arithmetic, not the production kernel's reduction order.
"""
from pathlib import Path
import math

import numpy as np

from .common import digest, read_json, write_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


# Output-only buffers are excluded from input comparisons. Read/write operands
# of fused_add_rms_norm remain inputs, and their post-call values are outputs.
OUTPUT_ONLY = {'_C.rms_norm.default': (0,), '_C.silu_and_mul.default': (0,)}
MUTATED = {'_C.rms_norm.default': (0,), '_C.silu_and_mul.default': (0,),
           '_C.fused_add_rms_norm.default': (0, 1)}
SUPPORTED = {
    'aten.mm.default', 'aten.bmm.default', 'aten.addmm.default',
    'aten.add.Tensor', 'aten.mul.Tensor', 'aten.div.Tensor',
    'aten.sum.dim_IntList', 'aten.mean.dim', 'aten.rsqrt.default',
    'aten.silu.default', 'aten.sigmoid.default', 'aten.pow.Tensor_Scalar',
    *MUTATED,
    'vllm.chunk_gated_delta_rule.forward_native',
    'vllm.rms_norm_gated.forward_cuda',
}


def gdn_reference(q, k, v, g, beta, initial_state=None, output_final_state=False,
                  cu_seqlens=None, use_qk_l2norm_in_kernel=True):
    """Sequential float64 gated-delta recurrence, independent of chunked kernels.

    S <- exp(g) S; S <- S + beta (v - S k) k^T; o <- S q / sqrt(K).
    State layout is [sequence, value_head, value_dim, key_dim].
    """
    import torch
    require(q.ndim == k.ndim == v.ndim == 4 and q.shape == k.shape, 'Unsupported GDN input layout')
    b, length, heads, kd = q.shape
    vh, vd = v.shape[-2:]
    require(v.shape[:2] == (b, length) and vh % heads == 0, 'Unsupported GDN head grouping')
    require(g.shape == beta.shape == (b, length, vh), 'GDN gates must be head-wise scalars')
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(vh // heads, dim=2) * kd**-.5
    k = k.repeat_interleave(vh // heads, dim=2)
    if cu_seqlens is None:
        spans = [(batch, 0, length) for batch in range(b)]
    else:
        starts = cu_seqlens.tolist()
        require(b == 1 and starts[0] == 0 and starts[-1] == length and
                all(a <= z for a, z in zip(starts, starts[1:])), 'Invalid variable-length GDN offsets')
        spans = [(0, a, z) for a, z in zip(starts, starts[1:])]
    if initial_state is not None:
        require(tuple(initial_state.shape) == (len(spans), vh, vd, kd), 'GDN initial-state shape mismatch')
    out = torch.empty_like(v, dtype=torch.float64)
    final = []
    for seq, (batch, start, end) in enumerate(spans):
        state = (initial_state[seq].clone() if initial_state is not None else
                 torch.zeros(vh, vd, kd, dtype=torch.float64))
        for t in range(start, end):
            qt, kt = q[batch, t], k[batch, t]
            state = state * g[batch, t].exp()[:, None, None]
            delta = (v[batch, t] - (state * kt[:, None, :]).sum(-1)) * beta[batch, t, :, None]
            state = state + delta[:, :, None] * kt[:, None, :]
            out[batch, t] = (state * qt[:, None, :]).sum(-1)
        final.append(state)
    return out, torch.stack(final) if output_final_state else None


class Archive:
    def __init__(self, max_bytes=512 * 1024**2):
        self.arrays = {}
        self.size = 0
        self.max_bytes = max_bytes

    def encode(self, value):
        import torch
        if isinstance(value, torch.Tensor):
            require(value.layout == torch.strided, 'Only dense tensors can be captured')
            # Copy NOW: many serving operations mutate/reuse their operands.
            cpu = value.detach().contiguous().cpu()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().copy().reshape(-1)
            self.size += raw.nbytes
            require(self.size <= self.max_bytes, 'Capture byte limit exceeded; narrow --module or increase --max-mib')
            name = f't{len(self.arrays):06d}'
            self.arrays[name] = raw
            return {'tensor': name, 'dtype': str(value.dtype).removeprefix('torch.'),
                    'shape': list(value.shape), 'stride': list(value.stride()),
                    'storage_offset': value.storage_offset(), 'sha256': digest(raw.tobytes())}
        if isinstance(value, tuple):
            return {'tuple': [self.encode(v) for v in value]}
        if isinstance(value, list):
            return {'list': [self.encode(v) for v in value]}
        if isinstance(value, dict):
            require(all(isinstance(k, str) for k in value), 'Only string dictionary keys are supported')
            return {'dict': {k: self.encode(v) for k, v in value.items()}}
        if isinstance(value, torch.dtype):
            return {'torch_dtype': str(value).removeprefix('torch.')}
        if isinstance(value, torch.device):
            return {'torch_device': str(value)}
        if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
            return value
        raise ValueError(f'Unsupported argument type: {type(value).__name__}')

    def save(self, path):
        np.savez(path, **self.arrays)
        return digest(Path(path).read_bytes())


def tensors(tree):
    if isinstance(tree, dict):
        if 'tensor' in tree:
            yield tree
        else:
            for v in tree.values():
                yield from tensors(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from tensors(v)


def logical_identity(tree):
    """Compare values/types/layout without file-local tensor identifiers."""
    if isinstance(tree, dict):
        if 'tensor' in tree:
            return {k: v for k, v in tree.items() if k not in ('tensor', 'storage_offset')}
        return {k: logical_identity(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [logical_identity(v) for v in tree]
    return tree


def validate_arrays(metadata, arrays):
    for t in tensors(metadata):
        require(t['tensor'] in arrays, f'Missing tensor {t["tensor"]}')
        raw = arrays[t['tensor']]
        require(raw.dtype == np.uint8 and raw.ndim == 1, 'Invalid tensor storage')
        require(digest(raw.tobytes()) == t['sha256'], f'Tensor changed: {t["tensor"]}')
        itemsize = {'bfloat16': 2, 'float16': 2, 'float32': 4, 'float64': 8,
                    'int64': 8, 'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1, 'bool': 1}.get(t['dtype'])
        require(itemsize is not None and raw.size == math.prod(t['shape']) * itemsize, 'Tensor shape/dtype mismatch')


def values(t, arrays):
    raw = arrays[t['tensor']]
    if t['dtype'] == 'bfloat16':
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(t['shape'])
    return raw.view(np.dtype(t['dtype'])).reshape(t['shape'])


def metrics(a, b, atol=0., rtol=0.):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.size > 0, 'Metrics require equal, nonempty shapes')
    x, y = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    err = np.abs(x[finite] - y[finite])
    close = np.zeros(x.shape, dtype=bool)
    close[finite] = err <= atol + rtol * np.abs(y[finite])
    return {'nonfinite_pairs': int((~finite).sum()),
            'max_absolute_error': float(err.max()) if err.size else None,
            'mean_absolute_error': float(err.mean()) if err.size else None,
            'relative_l2_error': float(np.linalg.norm(err) / max(np.linalg.norm(y[finite]), 1e-300)) if err.size else None,
            'within_tolerance_percent': float(close.mean() * 100),
            'within_tolerance': bool(close.all()), 'atol': atol, 'rtol': rtol}


def decode(tree, arrays, device='cpu', reference=False):
    import torch
    if isinstance(tree, dict):
        if 'tensor' in tree:
            dtype = getattr(torch, tree['dtype'])
            raw = torch.from_numpy(arrays[tree['tensor']].copy())
            src = raw.view(dtype).reshape(tree['shape'])
            if reference:
                return src.to(dtype=torch.float64 if src.is_floating_point() else src.dtype, device='cpu')
            # Preserve non-contiguous weights/activations, including transpose.
            shape, stride = tree['shape'], tree['stride']
            require(all(s >= 0 for s in stride), 'Negative strides cannot be replayed')
            span = 1 + sum((n - 1) * s for n, s in zip(shape, stride)) if src.numel() else 0
            offset = tree.get('storage_offset', 0)
            require(0 <= offset and span + offset <= max(src.numel() * 16, 4096),
                    'Excessive strided storage; unsupported replay layout')
            storage = torch.empty(span + offset, dtype=dtype, device=device)
            dst = storage.as_strided(shape, stride, offset)
            try:
                dst.copy_(src)
            except RuntimeError as e:
                raise ValueError('Overlapping/expanded tensor layout is not supported for replay') from e
            return dst
        if 'tuple' in tree:
            return tuple(decode(v, arrays, device, reference) for v in tree['tuple'])
        if 'list' in tree:
            return [decode(v, arrays, device, reference) for v in tree['list']]
        if 'dict' in tree:
            return {k: decode(v, arrays, device, reference) for k, v in tree['dict'].items()}
        if 'torch_dtype' in tree:
            return torch.float64 if reference and tree['torch_dtype'].startswith(('float', 'bfloat')) else getattr(torch, tree['torch_dtype'])
        if 'torch_device' in tree:
            return torch.device('cpu' if reference else device)
        if 'output_buffer' in tree:
            t = tree['output_buffer']
            return torch.empty_strided(t['shape'], t['stride'], dtype=getattr(torch, t['dtype']), device=device)
    return tree


def reference_op(name, args, kwargs):
    """CPU float64 arithmetic on EXACT captured (already quantized) operands."""
    import torch
    if name == 'vllm.chunk_gated_delta_rule.forward_native':
        return gdn_reference(*args, **kwargs)
    if name == 'vllm.rms_norm_gated.forward_cuda':
        x, weight, bias, z, eps, group_size, norm_before_gate, activation = args
        require(activation in ('swish', 'silu', 'sigmoid'), 'Unsupported gate activation')
        gate = None if z is None else (z.sigmoid() if activation == 'sigmoid' else z * z.sigmoid())
        if gate is not None and not norm_before_gate:
            x = x * gate
        group_size = group_size or x.shape[-1]
        grouped = x.reshape(*x.shape[:-1], -1, group_size)
        norm = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)).reshape_as(x)
        result = norm * weight
        if bias is not None:
            result = result + bias
        return result * gate if gate is not None and norm_before_gate else result
    if name in ('aten.mm.default', 'aten.bmm.default'):
        return args[0] @ args[1]
    if name == 'aten.addmm.default':
        return kwargs.get('beta', 1) * args[0] + kwargs.get('alpha', 1) * (args[1] @ args[2])
    if name == 'aten.add.Tensor':
        return args[0] + kwargs.get('alpha', 1) * args[1]
    if name == 'aten.mul.Tensor':
        return args[0] * args[1]
    if name == 'aten.div.Tensor':
        return args[0] / args[1]
    if name == 'aten.pow.Tensor_Scalar':
        return args[0] ** args[1]
    if name in ('aten.sum.dim_IntList', 'aten.mean.dim'):
        dims = args[1] if len(args) > 1 else kwargs.get('dim')
        dims = tuple(dims) if dims else None
        keep = args[2] if len(args) > 2 else kwargs.get('keepdim', False)
        return getattr(torch, 'sum' if 'sum.' in name else 'mean')(args[0], dim=dims, keepdim=keep)
    if name == 'aten.rsqrt.default':
        return torch.rsqrt(args[0])
    if name == 'aten.sigmoid.default':
        return torch.sigmoid(args[0])
    if name == 'aten.silu.default':
        return args[0] * torch.sigmoid(args[0])
    if name == '_C.rms_norm.default':
        _, x, weight, eps = args
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
    if name == '_C.fused_add_rms_norm.default':
        x, residual, weight, eps = args
        total = x + residual
        return (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps) * weight, total)
    if name == '_C.silu_and_mul.default':
        _, x = args
        a, b = x.chunk(2, -1)
        return a * torch.sigmoid(a) * b
    raise ValueError(f'No independent reference for {name}; add an explicit adapter')


def flatten_torch(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in flatten_torch(v)]
    return []


def replay(bundle, device, output, repeats=3, atol=0., rtol=0., profile=False):
    import torch
    root, output = Path(bundle), Path(output)
    require(not output.exists(), 'Replay output exists')
    require(repeats >= 2 and math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0,
            'Use at least two repeats and finite nonnegative tolerances')
    meta = read_json(root / 'operation.json')
    require(meta['schema_version'] == 1 and meta['operation'] in SUPPORTED, 'Unsupported operation bundle')
    require(digest({k: v for k, v in meta.items() if k != 'fingerprint'}) == meta['fingerprint'], 'Operation manifest changed')
    require(digest((root / 'tensors.npz').read_bytes()) == meta['tensors_sha256'], 'Operation archive changed')
    arrays = dict(np.load(root / 'tensors.npz', allow_pickle=False))
    validate_arrays(meta['event'], arrays)
    name, event = meta['operation'], meta['event']
    if not name.startswith('aten.'):
        require(device != 'cpu', 'This vLLM kernel requires a CUDA/HIP serving environment')
        import vllm._custom_ops  # Registers the actual installed production operators.
    if name == 'vllm.chunk_gated_delta_rule.forward_native':
        from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
        op = chunk_gated_delta_rule
    elif name == 'vllm.rms_norm_gated.forward_cuda':
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
        op = rmsnorm_fn
    else:
        namespace, opname, overload = name.split('.')
        op = getattr(getattr(getattr(torch.ops, namespace), opname), overload)
    args_ref = decode(event['args'], arrays, reference=True)
    kwargs_ref = decode(event['kwargs'], arrays, reference=True)
    # References do not need the uninitialized output-only buffer.
    with torch.inference_mode():
        ref = flatten_torch(reference_op(name, args_ref, kwargs_ref))
    captured = [values(t, arrays) for t in tensors(event['outputs'])]
    require(len(ref) == len(captured), 'Reference output structure mismatch')
    report = {'schema_version': 1, 'operation_fingerprint': meta['fingerprint'], 'operation': name,
              'environment': runtime_environment(device), 'reference': 'CPU float64, exact captured input values',
              'tolerance_note': 'User-specified numerical tolerance; bitwise mismatch alone does not establish a bug.',
              'repeats': [], 'input_layout_preserved': True, 'capture_execution': meta['capture_execution']}
    if name == 'vllm.chunk_gated_delta_rule.forward_native':
        report['reference_note'] = ('Sequential float64 recurrence versus a chunked low-precision implementation; '
                                    'normalization, chunk algebra and intermediate rounding may differ.')
    outputs = []
    output.mkdir(parents=True)
    for repeat in range(repeats):
        args = decode(event['args'], arrays, device)
        kwargs = decode(event['kwargs'], arrays, device)
        profiler = None
        if profile and repeat == repeats - 1:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device != 'cpu':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities, record_shapes=True)
            profiler.start()
        with torch.inference_mode():
            actual = op(*args, **kwargs)
            if name in MUTATED:
                actual = tuple(args[i] for i in MUTATED[name])
        if device != 'cpu':
            torch.cuda.synchronize()
        if profiler:
            profiler.stop()
            profiler.export_chrome_trace(str(output / 'kernel-trace.json'))
        actual = flatten_torch(actual)
        require(len(actual) == len(ref), 'Replay output structure mismatch')
        record, raw = [], []
        for i, (a, reference, saved) in enumerate(zip(actual, ref, captured)):
            a = a.detach().contiguous().cpu()
            bits = a.reshape(-1).view(torch.uint8).numpy().copy()
            raw.append(bits)
            record.append({'output': i, 'vs_float64': metrics(a.double().numpy(), reference.numpy(), atol, rtol),
                           'vs_captured': metrics(a.double().numpy(), saved, atol, rtol),
                           'bitwise_equal_to_capture': bool(np.array_equal(bits, arrays[list(tensors(event['outputs']))[i]['tensor']]))})
        outputs.append(raw)
        report['repeats'].append(record)
    report['repeat_bitwise_equal'] = all(all(np.array_equal(a, b) for a, b in zip(outputs[0], r)) for r in outputs[1:])
    report['all_within_tolerance'] = all(x['vs_float64']['within_tolerance'] for r in report['repeats'] for x in r)
    saved = {}
    for i, (a, reference) in enumerate(zip(actual, ref)):
        saved[f'output_{i}'] = a.detach().cpu().double().numpy()
        saved[f'output_{i}__bits'] = outputs[-1][i]
        saved[f'reference_{i}'] = reference.numpy()
    np.savez(output / 'outputs.npz', **saved)
    report['outputs_sha256'] = digest((output / 'outputs.npz').read_bytes())
    report['implementation_sha256'] = digest(Path(__file__).read_bytes())
    write_json(output / 'replay.json', report)
    return report


def runtime_environment(device='cuda'):
    import importlib.metadata
    import torch
    packages = {}
    for name in ('torch', 'vllm', 'triton', 'transformers'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {'device': torch.cuda.get_device_name() if device != 'cpu' else 'CPU',
            'cuda': torch.version.cuda, 'hip': torch.version.hip, 'packages': packages,
            'tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision()}
