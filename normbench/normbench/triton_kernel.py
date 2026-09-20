"""Diagnostic fused RMSNorm kernel with FP32 snapshots.

Formula and launch geometry follow vLLM 0.17.1 layernorm_guard.py (Apache-2.0,
original FlashAttention/FLA MIT implementation). Instrumentation may change
compiler decisions. Only use its intermediates when output/control checks pass.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def trace_norm(X, W, Z, Y, Square, Sum, Mean, EpsSum, Rstd, Norm, Weighted,
               Sigmoid, Gate, Precast, M, N: tl.constexpr, eps,
               BLOCK_N: tl.constexpr, ROWS: tl.constexpr, CAPTURE: tl.constexpr):
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
    inv = tl.rsqrt(variance_eps)
    normalized = x * inv[:, None]
    weighted = normalized * w[None, :]
    sigmoid = tl.sigmoid(z)
    gate = z * sigmoid
    precast = weighted * gate
    tl.store(Y + offset, precast, mask)
    # Store after computing the result; this still can alter compiler fusion.
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


def execute(backend, inputs, eps, capture):
    from .data import STAGES, SCALAR_STAGES
    x, w, z = [backend.upload(inputs[k], 'bfloat16') for k in ('x', 'w', 'z')]
    m, n = x.shape
    snapshots = {stage: torch.empty((m, 1) if stage in SCALAR_STAGES else (m, n),
                                   device=x.device, dtype=torch.bfloat16 if stage == 'output' else torch.float32)
                 for stage in (STAGES if capture else ('output',))}
    compute_units = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows = min(triton.next_power_of_2(triton.cdiv(m, 2 * compute_units)), 4)
    block = triton.next_power_of_2(n)
    trace_norm[(triton.cdiv(m, rows),)](
        x, w, z, snapshots['output'], *[snapshots.get(s) for s in STAGES[:-1]], m, n, eps,
        BLOCK_N=block, ROWS=rows, CAPTURE=capture, num_warps=min(max(block // 256, 1), 8))
    return snapshots


def instrument(backend, inputs, eps):
    return execute(backend, inputs, eps, capture=True)


def fused(backend, inputs, eps):
    return execute(backend, inputs, eps, capture=False)['output']
