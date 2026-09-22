"""Small GPU probe for the sigmoid example at (761, 11).

This reduced kernel can compile differently from the full benchmark.
Use reproduce_examples.py to check the original full-shape computation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def example(Z, FP32, BF16):
    z = tl.load(Z)
    sigmoid = tl.sigmoid(z)
    # This normalization/weight result agreed between A100 and MI210.
    weighted = 0.06220520660281181
    y = weighted * (z * sigmoid)
    tl.store(FP32, sigmoid)
    tl.store(FP32 + 1, y)
    tl.store(BF16, y)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemExit('Run inside an allocated GPU job with CUDA/ROCm PyTorch.')
    z = torch.tensor([-0.671875], dtype=torch.float32, device='cuda')
    fp32 = torch.empty(2, dtype=torch.float32, device='cuda')
    bf16 = torch.empty(1, dtype=torch.bfloat16, device='cuda')
    example[(1,)](z, fp32, bf16, num_warps=1)
    print('GPU:', torch.cuda.get_device_name())
    print('PyTorch:', torch.__version__, 'Triton:', triton.__version__)
    print('sigmoid(z):', fp32[0].item())
    print('before BF16:', fp32[1].item())
    print('after BF16: ', bf16.item())
    print('BF16 midpoint:', -0.014129638671875)
