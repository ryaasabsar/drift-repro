"""GPU microbench probe for numerical divergences between A100 and MI210.

Computes the three recorded divergence examples:
1. (761, 11)  - Sigmoid branch divergence
2. (1330, 36) - Sigmoid branch divergence
3. (1385, 8)  - Reciprocal square root (rsqrt) branch divergence

Also supports testing custom input numbers or running the comprehensive suite:
    python microbench_divergence.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_example(Z, Weighted, FP32, BF16):
    z = tl.load(Z)
    w = tl.load(Weighted)
    sigmoid = tl.sigmoid(z)
    y = w * (z * sigmoid)
    tl.store(FP32, sigmoid)
    tl.store(FP32 + 1, y)
    tl.store(BF16, y)


@triton.jit
def rsqrt_example(VarEps, X, W, Gate, FP32, BF16):
    vareps = tl.load(VarEps)
    x = tl.load(X)
    w = tl.load(W)
    gate = tl.load(Gate)
    inv = tl.rsqrt(vareps)
    norm = x * inv
    weighted = norm * w
    y = weighted * gate
    tl.store(FP32, inv)
    tl.store(FP32 + 1, y)
    tl.store(BF16, y)


def run_divergence_examples():
    if not torch.cuda.is_available():
        raise SystemExit('Run inside an allocated GPU job with CUDA/ROCm PyTorch.')

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"Platform: {'AMD ROCm' if torch.version.hip else 'NVIDIA CUDA'}\n")

    # Example 1: (761, 11) - Sigmoid
    print("=" * 60)
    print("Example 1: Coordinate (761, 11) - Sigmoid Divergence")
    print("=" * 60)
    z1 = torch.tensor([-0.671875], dtype=torch.float32, device='cuda')
    w1 = torch.tensor([0.06220520660281181], dtype=torch.float32, device='cuda')
    fp32_1 = torch.empty(2, dtype=torch.float32, device='cuda')
    bf16_1 = torch.empty(1, dtype=torch.bfloat16, device='cuda')
    sigmoid_example[(1,)](z1, w1, fp32_1, bf16_1, num_warps=1)
    print(f"Input z:            {z1.item()}")
    print(f"Input weighted:     {w1.item()}")
    print(f"sigmoid(z):         {fp32_1[0].item():.8f}")
    print(f"before BF16 (FP32): {fp32_1[1].item():.15f}")
    print(f"after BF16:         {bf16_1.item()}")
    print(f"BF16 midpoint:      -0.014129638671875")
    print("Recorded A100:      -0.01416015625 (0xbc68)")
    print("Recorded MI210:     -0.01409912109375 (0xbc67)")

    # Example 2: (1330, 36) - Sigmoid
    print("\n" + "=" * 60)
    print("Example 2: Coordinate (1330, 36) - Sigmoid Divergence")
    print("=" * 60)
    z2 = torch.tensor([-1.28125], dtype=torch.float32, device='cuda')
    w2 = torch.tensor([0.2603921592235565], dtype=torch.float32, device='cuda')
    fp32_2 = torch.empty(2, dtype=torch.float32, device='cuda')
    bf16_2 = torch.empty(1, dtype=torch.bfloat16, device='cuda')
    sigmoid_example[(1,)](z2, w2, fp32_2, bf16_2, num_warps=1)
    print(f"Input z:            {z2.item()}")
    print(f"Input weighted:     {w2.item()}")
    print(f"sigmoid(z):         {fp32_2[0].item():.8f}")
    print(f"before BF16 (FP32): {fp32_2[1].item():.15f}")
    print(f"after BF16:         {bf16_2.item()}")
    print(f"BF16 midpoint:      -0.072509765625")
    print("Recorded A100:      -0.072265625 (0xbd94)")
    print("Recorded MI210:     -0.07275390625 (0xbd95)")

    # Example 3: (1385, 8) - Rsqrt
    print("\n" + "=" * 60)
    print("Example 3: Coordinate (1385, 8) - Rsqrt Divergence")
    print("=" * 60)
    vareps3 = torch.tensor([7.3914202403102536e-06], dtype=torch.float32, device='cuda')
    x3 = torch.tensor([0.0003376007080078125], dtype=torch.float32, device='cuda')
    w3 = torch.tensor([0.921875], dtype=torch.float32, device='cuda')
    gate3 = torch.tensor([-0.10050322115421295], dtype=torch.float32, device='cuda')
    fp32_3 = torch.empty(2, dtype=torch.float32, device='cuda')
    bf16_3 = torch.empty(1, dtype=torch.bfloat16, device='cuda')
    rsqrt_example[(1,)](vareps3, x3, w3, gate3, fp32_3, bf16_3, num_warps=1)
    print(f"Input vareps:       {vareps3.item()}")
    print(f"rsqrt(vareps):      {fp32_3[0].item():.8f}")
    print(f"before BF16 (FP32): {fp32_3[1].item():.15f}")
    print(f"after BF16:         {bf16_3.item()}")
    print(f"BF16 midpoint:      -0.011505126953125")
    print("Recorded A100:      -0.01153564453125 (0xbc3d)")
    print("Recorded MI210:     -0.011474609375 (0xbc3c)")


if __name__ == '__main__':
    run_divergence_examples()
