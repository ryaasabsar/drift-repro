"""Frozen, model-free gated RMSNorm inputs and numerical comparisons.

Only NumPy is needed for fixture preparation and result comparison. BF16 rounding
is performed directly from float64, avoiding double rounding through float32.
"""
from pathlib import Path

import numpy as np

from .common import digest, read_json, write_json
from .numeric import metrics, require

FIXTURE = Path(__file__).resolve().parent / 'fixtures/gated_norm'
STAGES = ('square', 'sum_squares', 'mean_square', 'add_epsilon', 'rsqrt',
          'normalized', 'weighted', 'sigmoid', 'gate', 'precast', 'output')
SCALAR_STAGES = {'sum_squares', 'mean_square', 'add_epsilon', 'rsqrt'}
PARENTS = {
    'square': ('x',), 'sum_squares': ('square',), 'mean_square': ('sum_squares',),
    'add_epsilon': ('mean_square',), 'rsqrt': ('add_epsilon',),
    'normalized': ('x', 'rsqrt'), 'weighted': ('normalized', 'w'),
    'sigmoid': ('z',), 'gate': ('z', 'sigmoid'),
    'precast': ('weighted', 'gate'), 'output': ('precast',),
}


def bf16_bits(value):
    x = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(x)
    safe = np.where(finite, np.abs(x), 0.)
    _, exponent = np.frexp(safe)
    step = np.exp2(np.maximum(exponent - 8, -133).astype(np.float64))
    rounded = np.copysign(np.rint(safe / step) * step, x)
    rounded = np.where(finite, rounded, x)
    with np.errstate(over='ignore', invalid='ignore'):
        return (rounded.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_values(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def quantize(value, precision):
    return bf16_values(bf16_bits(value)) if precision == 'bfloat16' else np.asarray(value, dtype=np.float32)


def reference(stage, args, width, eps):
    a = [np.asarray(x, dtype=np.float64) for x in args]
    if stage == 'square':
        return a[0] * a[0]
    if stage == 'sum_squares':
        return a[0].sum(-1, keepdims=True)
    if stage == 'mean_square':
        return a[0] / width
    if stage == 'add_epsilon':
        return a[0] + eps
    if stage == 'rsqrt':
        return 1 / np.sqrt(a[0])
    if stage == 'sigmoid':
        return np.exp(-np.logaddexp(0., -a[0]))
    if stage == 'output':
        # Retain the unrounded value for absolute/L2 errors; ULP metrics round
        # this float64 value directly to the output's representable format.
        return a[0].copy()
    return a[0] * a[1]


def logical_bits(array, dtype):
    if dtype == 'bfloat16':
        return bf16_bits(array)
    require(dtype == 'float32', f'Unsupported result dtype: {dtype}')
    return np.asarray(array, dtype=np.float32).view(np.uint32)


def decode_bits(bits, dtype):
    return bf16_values(bits) if dtype == 'bfloat16' else bits.view(np.float32)


def compare_values(actual, expected, dtype):
    a, b = np.asarray(actual), np.asarray(expected)
    result = metrics(a, b)
    ab, bb = logical_bits(a, dtype), logical_bits(b, dtype)
    sign = 1 << (15 if dtype == 'bfloat16' else 31)
    def ordered(bits):
        raw = bits.astype(np.int64)
        return np.where((raw & sign) != 0, sign - (raw & (sign - 1)), sign + raw)
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(decode_bits(bb, dtype))
    distance = np.abs(ordered(ab)[finite] - ordered(bb)[finite])
    result.update(bitwise_equal=bool(np.array_equal(ab, bb)),
                  differing_elements=int(np.count_nonzero(ab != bb)), elements=int(a.size),
                  differing_percent=float(np.mean(ab != bb) * 100),
                  ulp_max=int(distance.max()) if distance.size else None,
                  ulp_p99=float(np.percentile(distance, 99)) if distance.size else None,
                  ulp_note=f'Distance to nearest {dtype} value; float64 reference rounded directly, ties to even.')
    return result


def make_fixture(output, inputs=None, eps=1e-6):
    """Freeze references from packaged inputs or a plain NPZ with x, w, z."""
    output = Path(output)
    require(not output.exists(), 'Fixture destination exists')
    require(np.isfinite(eps) and eps > 0, 'Epsilon must be finite and positive')
    if inputs is None:
        source, bundled = load_fixture()
        x, w, z = [bundled[f'captured/input/{key}'].copy() for key in ('x', 'w', 'z')]
        provenance = {'source_fixture_fingerprint': source['fingerprint']}
        focus = source['focus_positions']
    else:
        raw = dict(np.load(inputs, allow_pickle=False))
        require(set(raw) == {'x', 'w', 'z'}, 'Input NPZ must contain exactly x, w, z')
        x, w, z = [np.asarray(raw[key], dtype=np.float32) for key in ('x', 'w', 'z')]
        provenance = {'input_file_sha256': digest(Path(inputs).read_bytes())}
        focus = []
    require(x.ndim == 2 and z.shape == x.shape and x.size > 0, 'x and z must be equal, nonempty [rows, columns] tensors')
    require(w.shape in ((x.shape[1],), (1, x.shape[1])), 'w must contain one weight per column')
    require(all(np.isfinite(a).all() for a in (x, w, z)), 'Inputs must be finite')
    require(all(np.array_equal(a, quantize(a, 'bfloat16')) for a in (x, w, z)),
            'Inputs must already contain BF16-representable values; quantize explicitly before freezing')
    w = w.reshape(-1)
    rng = np.random.Generator(np.random.PCG64(42))
    rx, rz = rng.normal(size=(32, 128)), rng.uniform(-12, 12, size=(32, 128))
    rw = rng.uniform(.5, 1.5, size=(128,))
    cases = [('captured', x, w, z, 'Frozen supplied normalization tensors; original logical shape preserved.')]
    for scale in (.5, 1., 2.):
        cases.append((f'synthetic-scale-{scale:g}', quantize(rx * scale, 'bfloat16'),
                      quantize(rw, 'bfloat16'), quantize(rz, 'bfloat16'), 'Shared seeded inputs; only x scale changes.'))
    tiny = np.zeros((32, 128), dtype=np.float32)
    tiny[1::2] = 2.**-20
    cases.append(('epsilon-dominated', tiny, np.ones(128, dtype=np.float32),
                  quantize(rz, 'bfloat16'), 'Zero and tiny rows exercise epsilon and zero handling.'))
    arrays, entries = {}, []
    for name, cx, cw, cz, note in cases:
        base = {'x': cx.astype(np.float64), 'w': cw.reshape(1, -1).astype(np.float64),
                'z': cz.astype(np.float64)}
        for key, value in base.items():
            arrays[f'{name}/input/{key}'] = value.astype(np.float32)
        for stage in STAGES:
            base[stage] = reference(stage, [base[p] for p in PARENTS[stage]], cx.shape[-1], eps)
            arrays[f'{name}/reference/{stage}'] = base[stage]
            for precision in ('float32', 'bfloat16'):
                operands = [quantize(base[p], precision) for p in PARENTS[stage]]
                for i, operand in enumerate(operands):
                    arrays[f'{name}/isolated/{precision}/{stage}/arg{i}'] = operand
                # Scalar arguments obey the same staged precision protocol.
                scalar_eps = float(quantize(eps, precision))
                arrays[f'{name}/isolated/{precision}/{stage}/reference'] = reference(
                    stage, operands, cx.shape[-1], scalar_eps)
        entries.append({'name': name, 'kind': 'norm', 'shape': list(cx.shape), 'eps': eps, 'note': note})
    # FP32 values immediately below/at/above BF16 midpoints, both signs,
    # including exponent transitions near 0.5, 1, 2 and 4.
    anchors = np.array([.5, 1., 2., 4.], np.float32)
    lower = bf16_bits(anchors).astype(np.int32)
    probes = []
    for bits in np.concatenate([lower - 1, lower, lower + 1]):
        lo, hi = bf16_values(np.array([bits, bits + 1], np.uint16)).astype(np.float64)
        mid = np.float32((lo + hi) / 2)
        trio = np.array([np.nextafter(mid, -np.inf, dtype=np.float32), mid,
                         np.nextafter(mid, np.inf, dtype=np.float32)], np.float32)
        probes.extend(trio.tolist()); probes.extend((-trio).tolist())
    cast = np.tile(np.array(probes + [0., -0.], np.float32), 2).reshape(1, -1)
    arrays['bf16-midpoints/input/precast'] = cast
    arrays['bf16-midpoints/reference/output'] = cast.astype(np.float64)
    entries.append({'name': 'bf16-midpoints', 'kind': 'cast', 'shape': list(cast.shape),
                    'eps': eps, 'note': 'Exact FP32 midpoint and adjacent FP32 values; device FP32-to-BF16 cast only.'})
    output.mkdir(parents=True)
    unique, aliases, seen = {}, {}, {}
    for key, array in arrays.items():
        identity = (array.dtype.str, array.shape, digest(array.tobytes()))
        if identity not in seen:
            seen[identity] = f'array_{len(unique):04d}'
            unique[seen[identity]] = array
        aliases[key] = seen[identity]
    np.savez_compressed(output / 'cases.npz', **unique)
    manifest = {'schema_version': 1, 'cases': entries, 'stages': list(STAGES),
                'provenance': provenance, 'focus_positions': focus,
                'arrays_sha256': digest((output / 'cases.npz').read_bytes()), 'array_names': aliases,
                'reference': 'Frozen NumPy float64 arithmetic from exact BF16 input values; not an exact-real oracle.'}
    manifest['fingerprint'] = digest(manifest)
    write_json(output / 'fixture.json', manifest)
    return manifest


def load_fixture(path=FIXTURE):
    path = Path(path)
    manifest = read_json(path / 'fixture.json')
    require(manifest['schema_version'] == 1 and manifest['stages'] == list(STAGES), 'Unsupported fixture schema')
    require(digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) == manifest['fingerprint'], 'Changed fixture manifest')
    require(digest((path / 'cases.npz').read_bytes()) == manifest['arrays_sha256'], 'Changed fixture tensors')
    arrays = dict(np.load(path / 'cases.npz', allow_pickle=False))
    return manifest, {name: arrays[key] for name, key in manifest['array_names'].items()}
