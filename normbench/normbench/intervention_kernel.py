"""Controlled output substitutions in the standalone fused normalization kernel.

The original path calls the unchanged trace_norm JIT function. Shared-reference
and self-value controls use the same compiled substitution kernel; only loaded
tensor values differ. FP32 diagnostic stores still require observer checks.
"""
import torch
import triton
import triton.language as tl

from .data import STAGES, SCALAR_STAGES
from .triton_kernel import trace_norm


@triton.jit
def substituted_norm(X, W, Z, SharedInv, SharedSig, Switches, Y, Square, Sum, Mean,
                     EpsSum, Rstd, Norm, Weighted, Sigmoid, Gate, Precast,
                     M, N: tl.constexpr, eps, BLOCK_N: tl.constexpr,
                     ROWS: tl.constexpr, CAPTURE: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offset = rows[:, None] * N + cols[None, :]
    x = tl.load(X + offset, mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, cols < N, other=0.).to(tl.float32)
    z = tl.load(Z + offset, mask, other=0.).to(tl.float32)
    square = x * x
    total = tl.sum(square, axis=1)
    mean = total / N
    variance_eps = mean + eps
    # Device data selects the intervention. All arms retain identical compiled
    # instructions, including the original operations, and the same launch.
    inv = tl.where(tl.load(Switches) != 0,
                   tl.load(SharedInv + rows, rows < M, other=0.), tl.rsqrt(variance_eps))
    normalized = x * inv[:, None]
    weighted = normalized * w[None, :]
    sigmoid = tl.where(tl.load(Switches + 1) != 0,
                       tl.load(SharedSig + offset, mask, other=0.), tl.sigmoid(z))
    gate = z * sigmoid
    precast = weighted * gate
    tl.store(Y + offset, precast, mask)
    if CAPTURE:
        tl.store(Square + offset, square, mask)
        tl.store(Sum + rows, total, rows < M)
        tl.store(Mean + rows, mean, rows < M)
        tl.store(EpsSum + rows, variance_eps, rows < M)
        tl.store(Rstd + rows, inv, rows < M)
        tl.store(Norm + offset, normalized, mask)
        tl.store(Weighted + offset, weighted, mask)
        tl.store(Sigmoid + offset, sigmoid, mask)
        tl.store(Gate + offset, gate, mask)
        tl.store(Precast + offset, precast, mask)


def launch(inputs, replacements, eps, capture, switches=None):
    """Return device snapshots, the actual compiled kernel, and launch geometry."""
    x, w, z = [inputs[k] for k in ('x', 'w', 'z')]
    m, n = x.shape
    snapshots = {stage: torch.empty((m, 1) if stage in SCALAR_STAGES else (m, n),
                                   device=x.device, dtype=torch.bfloat16 if stage == 'output' else torch.float32)
                 for stage in (STAGES if capture else ('output',))}
    units = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows = min(triton.next_power_of_2(triton.cdiv(m, 2 * units)), 4)
    block = triton.next_power_of_2(n)
    warps = min(max(block // 256, 1), 8)
    grid = (triton.cdiv(m, rows),)
    args = [snapshots['output'], *[snapshots.get(s) for s in STAGES[:-1]], m, n, eps]
    options = dict(BLOCK_N=block, ROWS=rows, CAPTURE=capture, num_warps=warps)
    if switches is not None:
        kernel = substituted_norm[grid](
            x, w, z, replacements['rsqrt'], replacements['sigmoid'], switches, *args, **options)
    else:
        kernel = trace_norm[grid](x, w, z, *args, **options)
    return snapshots, kernel, {'grid': list(grid), 'rows_per_block': rows, 'block_n': block, 'num_warps': warps}
