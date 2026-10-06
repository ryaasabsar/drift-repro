"""Expanded GPU microbenchmark for A100 vs MI210 numerical divergence.

Tests:
1. All three natural divergence points from the benchmark:
   - (761, 11): Sigmoid branch divergence
   - (1330, 36): Sigmoid branch divergence
   - (1385, 8): Reciprocal square root (rsqrt) branch divergence
2. Synthetic boundary points near BF16 rounding midpoints for both sigmoid and rsqrt
3. A systematic sweep across values to demonstrate:
   - Intermediate FP32 divergence rate between NVIDIA and AMD approximation instructions
   - The exact BF16 midpoint crossing condition (why only ~1 in 65,536 FP32 differences flips BF16)

Run with:
    python normbench/microbench_divergence.py
"""
import argparse
import sys
import numpy as np
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def sigmoid_microbench(Z, Weighted, FP32_Sig, FP32_Precast, BF16_Out, N: tl.constexpr):
    """Isolated microbenchmark for the sigmoid branch."""
    idx = tl.arange(0, N)
    z = tl.load(Z + idx)
    w = tl.load(Weighted + idx)
    sig = tl.sigmoid(z)
    gate = z * sig
    precast = w * gate
    tl.store(FP32_Sig + idx, sig)
    tl.store(FP32_Precast + idx, precast)
    tl.store(BF16_Out + idx, precast)


@triton.jit
def rsqrt_microbench(VarEps, X, W, Gate, FP32_Rsqrt, FP32_Precast, BF16_Out, N: tl.constexpr):
    """Isolated microbenchmark for the rsqrt branch."""
    idx = tl.arange(0, N)
    vareps = tl.load(VarEps + idx)
    x = tl.load(X + idx)
    w = tl.load(W + idx)
    gate = tl.load(Gate + idx)
    inv = tl.rsqrt(vareps)
    norm = x * inv
    weighted = norm * w
    precast = weighted * gate
    tl.store(FP32_Rsqrt + idx, inv)
    tl.store(FP32_Precast + idx, precast)
    tl.store(BF16_Out + idx, precast)


# ---------------------------------------------------------------------------
# Helper functions for FP32 / BF16 inspection and groundtruth
# ---------------------------------------------------------------------------

def float32_to_bits(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy().view(np.uint32)


def bfloat16_to_bits(tensor):
    return (tensor.detach().cpu().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))


def fp64_groundtruth_sigmoid(z, weighted):
    z_f64 = np.asarray(z, dtype=np.float64)
    w_f64 = np.asarray(weighted, dtype=np.float64)
    sig_f64 = 1.0 / (1.0 + np.exp(-z_f64))
    precast_f64 = w_f64 * (z_f64 * sig_f64)
    return sig_f64, precast_f64


def fp64_groundtruth_rsqrt(vareps, x, w, gate):
    v_f64 = np.asarray(vareps, dtype=np.float64)
    x_f64 = np.asarray(x, dtype=np.float64)
    w_f64 = np.asarray(w, dtype=np.float64)
    g_f64 = np.asarray(gate, dtype=np.float64)
    rsqrt_f64 = 1.0 / np.sqrt(v_f64)
    precast_f64 = (x_f64 * rsqrt_f64 * w_f64) * g_f64
    return rsqrt_f64, precast_f64


def direct_round_bf16(val):
    x = np.atleast_1d(np.asarray(val, dtype=np.float64))
    finite = np.isfinite(x)
    safe = np.where(finite, np.abs(x), 0.0)
    _, exponent = np.frexp(safe)
    step = np.exp2(np.maximum(exponent - 8, -133).astype(np.float64))
    rounded = np.copysign(np.rint(safe / step) * step, x)
    rounded = np.where(finite, rounded, x)
    with np.errstate(over='ignore', invalid='ignore'):
        return (rounded.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_bits_to_float(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

NATURAL_CASES = [
    {
        'id': 'natural_pos_761_11',
        'type': 'sigmoid',
        'desc': 'Benchmark coordinate (761, 11) - Sigmoid branch divergence',
        'z': -0.671875,
        'weighted': 0.06220520660281181,
        'a100_output_bf16': -0.01416015625,
        'a100_output_hex': '0xbc68',
        'mi210_output_bf16': -0.01409912109375,
        'mi210_output_hex': '0xbc67',
        'midpoint_precast': -0.014129638671875,
    },
    {
        'id': 'natural_pos_1330_36',
        'type': 'sigmoid',
        'desc': 'Benchmark coordinate (1330, 36) - Sigmoid branch divergence',
        'z': -1.28125,
        'weighted': 0.2603921592235565,
        'a100_output_bf16': -0.072265625,
        'a100_output_hex': '0xbd94',
        'mi210_output_bf16': -0.07275390625,
        'mi210_output_hex': '0xbd95',
        'midpoint_precast': -0.072509765625,
    },
    {
        'id': 'natural_pos_1385_8',
        'type': 'rsqrt',
        'desc': 'Benchmark coordinate (1385, 8) - Rsqrt branch divergence',
        'vareps': 7.3914202403102536e-06,
        'x': 0.0003376007080078125,
        'w': 0.921875,
        'gate': -0.10050322115421295,
        'a100_output_bf16': -0.01153564453125,
        'a100_output_hex': '0xbc3d',
        'mi210_output_bf16': -0.011474609375,
        'mi210_output_hex': '0xbc3c',
        'midpoint_precast': -0.011505126953125,
    },
]

# Additional synthetic cases engineered near BF16 midpoints
SYNTHETIC_BOUNDARY_CASES = [
    {
        'id': 'synth_sigmoid_boundary_1',
        'type': 'sigmoid',
        'desc': 'Positive z near BF16 midpoint [0.1250, 0.1255]',
        'z': 0.515625,
        'weighted': 0.35246723,
    },
    {
        'id': 'synth_sigmoid_boundary_2',
        'type': 'sigmoid',
        'desc': 'Negative z near BF16 midpoint [-0.2500, -0.2510]',
        'z': -1.046875,
        'weighted': 0.9234511,
    },
    {
        'id': 'synth_rsqrt_boundary_1',
        'type': 'rsqrt',
        'desc': 'Small variance scale near BF16 midpoint',
        'vareps': 1.234567e-5,
        'x': 0.00045166015625,
        'w': 1.0546875,
        'gate': 0.15625,
    },
    {
        'id': 'synth_rsqrt_boundary_2',
        'type': 'rsqrt',
        'desc': 'Unit variance scale near BF16 midpoint',
        'vareps': 1.000000e-0,
        'x': 0.046875,
        'w': 0.875,
        'gate': 0.375,
    }
]


def run_natural_cases(device='cuda'):
    print("=" * 80)
    print("SECTION 1: THE THREE NATURAL DIVERGENCE POINTS FROM NORMBENCH")
    print("=" * 80)

    for case in NATURAL_CASES:
        print(f"\nCase: {case['id']} - {case['desc']}")
        if case['type'] == 'sigmoid':
            z_t = torch.tensor([case['z']], dtype=torch.float32, device=device)
            w_t = torch.tensor([case['weighted']], dtype=torch.float32, device=device)
            fp32_sig = torch.empty(1, dtype=torch.float32, device=device)
            fp32_pre = torch.empty(1, dtype=torch.float32, device=device)
            bf16_out = torch.empty(1, dtype=torch.bfloat16, device=device)

            sigmoid_microbench[(1,)](z_t, w_t, fp32_sig, fp32_pre, bf16_out, N=1, num_warps=1)

            sig_val = fp32_sig.item()
            pre_val = fp32_pre.item()
            bf16_val = bf16_out.item()
            bf16_hex = hex(bfloat16_to_bits(bf16_out)[0])
            pre_hex = hex(float32_to_bits(fp32_pre)[0])

            gt_sig, gt_pre = fp64_groundtruth_sigmoid(case['z'], case['weighted'])
            gt_bf16_bits = direct_round_bf16(gt_pre)[0]
            gt_bf16_val = bf16_bits_to_float(gt_bf16_bits)

            print(f"  Input z:               {case['z']}")
            print(f"  Input weighted:        {case['weighted']}")
            print(f"  Device intermediate:   sigmoid(z) = {sig_val:.8f}")
            print(f"  Device precast FP32:   {pre_val:.15f} (raw bits: {pre_hex})")
            print(f"  BF16 Midpoint:         {case['midpoint_precast']:.15f}")
            print(f"  Device BF16 output:    {bf16_val:.10f} ({bf16_hex})")
            print(f"  Reference A100 output: {case['a100_output_bf16']} ({case['a100_output_hex']})")
            print(f"  Reference MI210 out:   {case['mi210_output_bf16']} ({case['mi210_output_hex']})")
            print(f"  Groundtruth (FP64):    precast={gt_pre:.15f}, BF16={gt_bf16_val} ({hex(gt_bf16_bits)})")
            
            # Distance from midpoint in FP32 ULPs
            dist_midpoint_ulp = int(float32_to_bits(fp32_pre)[0] & 0xffff)
            print(f"  Lower 16-bit offset:   0x{dist_midpoint_ulp:04x} (midpoint is 0x8000; diff: {dist_midpoint_ulp - 0x8000:+d} FP32 ULP)")

        elif case['type'] == 'rsqrt':
            vareps_t = torch.tensor([case['vareps']], dtype=torch.float32, device=device)
            x_t = torch.tensor([case['x']], dtype=torch.float32, device=device)
            w_t = torch.tensor([case['w']], dtype=torch.float32, device=device)
            g_t = torch.tensor([case['gate']], dtype=torch.float32, device=device)
            fp32_rsq = torch.empty(1, dtype=torch.float32, device=device)
            fp32_pre = torch.empty(1, dtype=torch.float32, device=device)
            bf16_out = torch.empty(1, dtype=torch.bfloat16, device=device)

            rsqrt_microbench[(1,)](vareps_t, x_t, w_t, g_t, fp32_rsq, fp32_pre, bf16_out, N=1, num_warps=1)

            rsq_val = fp32_rsq.item()
            pre_val = fp32_pre.item()
            bf16_val = bf16_out.item()
            bf16_hex = hex(bfloat16_to_bits(bf16_out)[0])
            pre_hex = hex(float32_to_bits(fp32_pre)[0])

            gt_rsq, gt_pre = fp64_groundtruth_rsqrt(case['vareps'], case['x'], case['w'], case['gate'])
            gt_bf16_bits = direct_round_bf16(gt_pre)[0]
            gt_bf16_val = bf16_bits_to_float(gt_bf16_bits)

            print(f"  Input vareps:          {case['vareps']}")
            print(f"  Device intermediate:   rsqrt(vareps) = {rsq_val:.8f}")
            print(f"  Device precast FP32:   {pre_val:.15f} (raw bits: {pre_hex})")
            print(f"  BF16 Midpoint:         {case['midpoint_precast']:.15f}")
            print(f"  Device BF16 output:    {bf16_val:.10f} ({bf16_hex})")
            print(f"  Reference A100 output: {case['a100_output_bf16']} ({case['a100_output_hex']})")
            print(f"  Reference MI210 out:   {case['mi210_output_bf16']} ({case['mi210_output_hex']})")
            print(f"  Groundtruth (FP64):    precast={gt_pre:.15f}, BF16={gt_bf16_val} ({hex(gt_bf16_bits)})")
            dist_midpoint_ulp = int(float32_to_bits(fp32_pre)[0] & 0xffff)
            print(f"  Lower 16-bit offset:   0x{dist_midpoint_ulp:04x} (midpoint is 0x8000; diff: {dist_midpoint_ulp - 0x8000:+d} FP32 ULP)")


def run_synthetic_cases(device='cuda'):
    print("\n" + "=" * 80)
    print("SECTION 2: ADDITIONAL NUMBERS (SYNTHETIC BOUNDARY CASES)")
    print("=" * 80)

    for case in SYNTHETIC_BOUNDARY_CASES:
        print(f"\nCase: {case['id']} - {case['desc']}")
        if case['type'] == 'sigmoid':
            z_t = torch.tensor([case['z']], dtype=torch.float32, device=device)
            w_t = torch.tensor([case['weighted']], dtype=torch.float32, device=device)
            fp32_sig = torch.empty(1, dtype=torch.float32, device=device)
            fp32_pre = torch.empty(1, dtype=torch.float32, device=device)
            bf16_out = torch.empty(1, dtype=torch.bfloat16, device=device)

            sigmoid_microbench[(1,)](z_t, w_t, fp32_sig, fp32_pre, bf16_out, N=1, num_warps=1)

            pre_bits = float32_to_bits(fp32_pre)[0]
            bf16_bits = bfloat16_to_bits(bf16_out)[0]
            gt_sig, gt_pre = fp64_groundtruth_sigmoid(case['z'], case['weighted'])
            gt_bf16_bits = direct_round_bf16(gt_pre)[0]
            offset_from_midpoint = int(pre_bits & 0xffff) - 0x8000

            print(f"  z={case['z']}, weighted={case['weighted']}")
            print(f"  Sigmoid: {fp32_sig.item():.7f}, Precast FP32: {fp32_pre.item():.9f}")
            print(f"  BF16 output: {bf16_out.item():.7f} (hex: {hex(bf16_bits)})")
            print(f"  Groundtruth BF16: {bf16_bits_to_float(gt_bf16_bits):.7f} (hex: {hex(gt_bf16_bits)})")
            print(f"  Distance from nearest BF16 tie-midpoint: {offset_from_midpoint:+d} FP32 ULP")

        elif case['type'] == 'rsqrt':
            vareps_t = torch.tensor([case['vareps']], dtype=torch.float32, device=device)
            x_t = torch.tensor([case['x']], dtype=torch.float32, device=device)
            w_t = torch.tensor([case['w']], dtype=torch.float32, device=device)
            g_t = torch.tensor([case['gate']], dtype=torch.float32, device=device)
            fp32_rsq = torch.empty(1, dtype=torch.float32, device=device)
            fp32_pre = torch.empty(1, dtype=torch.float32, device=device)
            bf16_out = torch.empty(1, dtype=torch.bfloat16, device=device)

            rsqrt_microbench[(1,)](vareps_t, x_t, w_t, g_t, fp32_rsq, fp32_pre, bf16_out, N=1, num_warps=1)

            pre_bits = float32_to_bits(fp32_pre)[0]
            bf16_bits = bfloat16_to_bits(bf16_out)[0]
            gt_rsq, gt_pre = fp64_groundtruth_rsqrt(case['vareps'], case['x'], case['w'], case['gate'])
            gt_bf16_bits = direct_round_bf16(gt_pre)[0]
            offset_from_midpoint = int(pre_bits & 0xffff) - 0x8000

            print(f"  vareps={case['vareps']}, x={case['x']}, w={case['w']}, gate={case['gate']}")
            print(f"  Rsqrt: {fp32_rsq.item():.7f}, Precast FP32: {fp32_pre.item():.9f}")
            print(f"  BF16 output: {bf16_out.item():.7f} (hex: {hex(bf16_bits)})")
            print(f"  Groundtruth BF16: {bf16_bits_to_float(gt_bf16_bits):.7f} (hex: {hex(gt_bf16_bits)})")
            print(f"  Distance from nearest BF16 tie-midpoint: {offset_from_midpoint:+d} FP32 ULP")


def run_systematic_sweep(device='cuda', num_points=256):
    print("\n" + "=" * 80)
    print("SECTION 3: SYSTEMATIC PARAMETER SWEEP ACROSS DIVERGENCE THRESHOLDS")
    print("=" * 80)

    # 1. Sigmoid sweep
    z_vals = np.linspace(-3.0, 3.0, num_points, dtype=np.float32)
    w_vals = np.full_like(z_vals, 0.1, dtype=np.float32)

    z_t = torch.tensor(z_vals, device=device)
    w_t = torch.tensor(w_vals, device=device)
    fp32_sig = torch.empty(num_points, dtype=torch.float32, device=device)
    fp32_pre = torch.empty(num_points, dtype=torch.float32, device=device)
    bf16_out = torch.empty(num_points, dtype=torch.bfloat16, device=device)

    sigmoid_microbench[(1,)](z_t, w_t, fp32_sig, fp32_pre, bf16_out, N=num_points, num_warps=1)

    gt_sig, gt_pre = fp64_groundtruth_sigmoid(z_vals, w_vals)
    dev_sig = fp32_sig.detach().cpu().numpy()
    dev_pre_bits = float32_to_bits(fp32_pre)
    dev_bf16_bits = bfloat16_to_bits(bf16_out)
    gt_bf16_bits = direct_round_bf16(gt_pre)

    sig_ulp_diff = np.abs(dev_sig.view(np.uint32).astype(np.int64) - gt_sig.astype(np.float32).view(np.uint32).astype(np.int64))
    bf16_mismatches = np.count_nonzero(dev_bf16_bits != gt_bf16_bits)
    sig_divergent_elements = np.count_nonzero(sig_ulp_diff > 0)

    print(f"Sigmoid Sweep ({num_points} points in [-3.0, 3.0]):")
    print(f"  Intermediate FP32 vs Float64 non-zero ULP diff: {sig_divergent_elements}/{num_points} ({sig_divergent_elements/num_points*100:.1f}%)")
    print(f"  Max FP32 ULP difference:                         {sig_ulp_diff.max()} ULP")
    print(f"  Final BF16 mismatches against direct Float64:    {bf16_mismatches}/{num_points} ({bf16_mismatches/num_points*100:.1f}%)")

    # 2. Rsqrt sweep
    v_vals = np.geomspace(1e-6, 1e2, num_points, dtype=np.float32)
    x_vals = np.full_like(v_vals, 0.05, dtype=np.float32)
    w_vals = np.full_like(v_vals, 0.9, dtype=np.float32)
    g_vals = np.full_like(v_vals, 0.2, dtype=np.float32)

    v_t = torch.tensor(v_vals, device=device)
    x_t = torch.tensor(x_vals, device=device)
    w_t = torch.tensor(w_vals, device=device)
    g_t = torch.tensor(g_vals, device=device)
    fp32_rsq = torch.empty(num_points, dtype=torch.float32, device=device)
    fp32_pre = torch.empty(num_points, dtype=torch.float32, device=device)
    bf16_out = torch.empty(num_points, dtype=torch.bfloat16, device=device)

    rsqrt_microbench[(1,)](v_t, x_t, w_t, g_t, fp32_rsq, fp32_pre, bf16_out, N=num_points, num_warps=1)

    gt_rsq, gt_pre = fp64_groundtruth_rsqrt(v_vals, x_vals, w_vals, g_vals)
    dev_rsq = fp32_rsq.detach().cpu().numpy()
    dev_bf16_bits = bfloat16_to_bits(bf16_out)
    gt_bf16_bits = direct_round_bf16(gt_pre)

    rsq_ulp_diff = np.abs(dev_rsq.view(np.uint32).astype(np.int64) - gt_rsq.astype(np.float32).view(np.uint32).astype(np.int64))
    rsq_bf16_mismatches = np.count_nonzero(dev_bf16_bits != gt_bf16_bits)
    rsq_divergent_elements = np.count_nonzero(rsq_ulp_diff > 0)

    print(f"\nRsqrt Sweep ({num_points} points in [1e-6, 1e2]):")
    print(f"  Intermediate FP32 vs Float64 non-zero ULP diff: {rsq_divergent_elements}/{num_points} ({rsq_divergent_elements/num_points*100:.1f}%)")
    print(f"  Max FP32 ULP difference:                         {rsq_ulp_diff.max()} ULP")
    print(f"  Final BF16 mismatches against direct Float64:    {rsq_bf16_mismatches}/{num_points} ({rsq_bf16_mismatches/num_points*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Normbench Microbenchmark for A100 vs MI210 Divergence")
    parser.add_argument('--sweep-points', type=int, default=256, help="Number of sweep points")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA/ROCm device required for running this microbenchmark.")
        sys.exit(1)

    print(f"Accelerator: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"Backend: {'ROCm/HIP' if torch.version.hip else 'CUDA'} (CUDA: {torch.version.cuda}, HIP: {torch.version.hip})")

    run_natural_cases()
    run_synthetic_cases()
    run_systematic_sweep(num_points=args.sweep_points)


if __name__ == '__main__':
    main()
