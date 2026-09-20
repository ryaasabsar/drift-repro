"""Accelerator operations; no device work or accelerator imports at module load."""
import importlib.metadata
import inspect
import os

import numpy as np

from .common import digest
from .numeric import require


def packages():
    result = {}
    for name in ('numpy', 'torch', 'triton', 'vllm', 'ttnn'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


class TorchBackend:
    def __init__(self, backend, control='native'):
        import torch
        self.torch = torch
        self.device = 'cpu' if backend == 'cpu' else 'cuda'
        if backend != 'cpu':
            require(torch.cuda.is_available(), 'No allocated accelerator is accessible in this Python environment')
            require(bool(torch.version.hip) == (backend == 'amd'), 'PyTorch runtime does not match selected NVIDIA/AMD backend')
            torch.ones(1, device='cuda').add_(1).item()
        torch.set_num_threads(4)
        self.info = {'backend': backend, 'device': 'CPU' if backend == 'cpu' else torch.cuda.get_device_name(),
                     'cuda': torch.version.cuda, 'hip': torch.version.hip, 'packages': packages(),
                     'tf32': torch.backends.cuda.matmul.allow_tf32,
                     'float32_matmul_precision': torch.get_float32_matmul_precision(),
                     'staged_implementation': 'PyTorch eager, one operator per stage',
                     'visibility': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                            'ROCR_VISIBLE_DEVICES') if k in os.environ}}
        self.native_info = {'status': 'unavailable', 'reason': 'CPU has no production GPU control'}
        self.native_fn = None
        if backend != 'cpu':
            if control == 'vllm':
                # Explicit opt-in. Default runs never import a serving framework.
                from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
                from vllm.model_executor.layers.fla.ops import layernorm_guard
                self.native_fn = rmsnorm_fn
                self.native_info = {'status': 'available', 'implementation': 'vllm.rmsnorm_fn',
                                    'source_sha256': digest(inspect.getsource(layernorm_guard).encode())}
            else:
                from . import triton_kernel
                self.native_info = {'status': 'available', 'implementation': 'NormBench fused Triton kernel, snapshots disabled',
                                    'source_sha256': digest(inspect.getsource(triton_kernel).encode())}

    def upload(self, array, precision):
        return self.torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(
            device=self.device, dtype=getattr(self.torch, precision))

    def download(self, tensor):
        return tensor.detach().float().cpu().numpy().copy()

    def dtype(self, tensor):
        return str(tensor.dtype).removeprefix('torch.')

    def operation(self, stage, args, width, eps):
        t = self.torch
        x = args[0]
        if stage == 'square':
            return x * x
        if stage == 'sum_squares':
            return x.sum(-1, keepdim=True)
        if stage == 'mean_square':
            return x * (1. / width)
        if stage == 'add_epsilon':
            return x + eps
        if stage == 'rsqrt':
            return t.rsqrt(x)
        if stage == 'sigmoid':
            return t.sigmoid(x)
        if stage == 'output':
            return x.to(t.bfloat16)
        return x * args[1]

    def native(self, inputs, eps):
        if self.device == 'cpu':
            return None
        if self.native_fn is None:
            from .triton_kernel import fused
            return fused(self, inputs, eps)
        x, w, z = [self.upload(inputs[k], 'bfloat16') for k in ('x', 'w', 'z')]
        with self.torch.inference_mode():
            return self.native_fn(x, w.reshape(-1), None, z, eps, None, True, 'swish')

    def instrument(self, inputs, eps):
        if self.device == 'cpu':
            return None
        from .triton_kernel import instrument
        return instrument(self, inputs, eps)

    def close(self):
        if self.device != 'cpu':
            self.torch.cuda.synchronize()


class TTBackend:
    """TTNN 0.77 API, real device execution with explicit precision checks.

RMSNorm plus a separate gate is a native TTNN composition, not a claim to be
the exact fused kernel used by a particular Tenstorrent model implementation.
"""
    def __init__(self, device_id):
        import torch
        import ttnn
        self.torch, self.tt = torch, ttnn
        self.device = ttnn.open_device(device_id=device_id)
        try:
            arch = str(self.device.arch())
            require(any(name in arch.lower() for name in ('blackhole', 'wormhole')), 'This TTNN adapter supports Blackhole and Wormhole')
            config = ttnn.BlackholeComputeKernelConfig if 'blackhole' in arch.lower() else ttnn.WormholeComputeKernelConfig
            self.compute_config = config(math_fidelity=ttnn.MathFidelity.HiFi4,
                                         math_approx_mode=False, fp32_dest_acc_en=True)
            self.info = {'backend': 'tenstorrent', 'device': arch, 'device_id': device_id,
                         'packages': packages(), 'layout': 'TILE, interleaved DRAM, logical 4D',
                         'compute_config': {'math_fidelity': 'HiFi4', 'math_approx_mode': False,
                                            'fp32_dest_acc_en': True, 'applies_to': ['sum', 'rms_norm']},
                         'sigmoid': 'ttnn.sigmoid_accurate(fast_and_approximate_mode=False)',
                         'rsqrt': 'ttnn.rsqrt(fast_and_approximate_mode=False)',
                         'visibility': {k: os.environ[k] for k in ('TT_VISIBLE_DEVICES', 'TT_METAL_VISIBLE_DEVICES') if k in os.environ}}
            self.native_info = {'status': 'available', 'implementation': 'TTNN FP32 RMSNorm + accurate SiLU gate + BF16 cast',
                                'note': 'Native composition; no equivalence claim to vLLM-TT model kernels.'}
        except BaseException:
            ttnn.close_device(self.device)
            raise

    def upload(self, array, precision):
        array = np.asarray(array, dtype=np.float32)
        require(array.ndim == 2, 'TT inputs must have logical [rows, columns] shape')
        return self.tt.from_torch(self.torch.from_numpy(array.copy()).reshape(1, 1, *array.shape),
                                  dtype=getattr(self.tt, precision), layout=self.tt.TILE_LAYOUT,
                                  device=self.device, memory_config=self.tt.DRAM_MEMORY_CONFIG)

    def download(self, tensor):
        self.tt.synchronize_device(self.device)
        t = self.tt.to_torch(tensor).float()
        require(t.ndim == 4 and t.shape[:2] == (1, 1), 'Unexpected TTNN logical output shape')
        return t[0, 0].numpy().copy()

    def dtype(self, tensor):
        if tensor.dtype == self.tt.float32:
            return 'float32'
        if tensor.dtype == self.tt.bfloat16:
            return 'bfloat16'
        raise ValueError(f'Unexpected TTNN result dtype: {tensor.dtype}')

    def operation(self, stage, args, width, eps):
        t, x = self.tt, args[0]
        if stage == 'sum_squares':
            return t.sum(x, dim=-1, keepdim=True, compute_kernel_config=self.compute_config)
        if stage == 'mean_square':
            return t.multiply(x, 1. / width)
        if stage == 'add_epsilon':
            return t.add(x, eps)
        if stage == 'rsqrt':
            return t.rsqrt(x, fast_and_approximate_mode=False)
        if stage == 'sigmoid':
            return t.sigmoid_accurate(x, fast_and_approximate_mode=False)
        if stage == 'output':
            return t.typecast(x, t.bfloat16)
        return t.multiply(x, x if stage == 'square' else args[1])

    def native(self, inputs, eps):
        t = self.tt
        x, w, z = [self.upload(inputs[k], 'float32') for k in ('x', 'w', 'z')]
        normalized = t.rms_norm(x, weight=w, epsilon=eps, compute_kernel_config=self.compute_config)
        gate = t.multiply(z, t.sigmoid_accurate(z, fast_and_approximate_mode=False))
        return t.typecast(t.multiply(normalized, gate), t.bfloat16)

    def instrument(self, inputs, eps):
        return None  # Triton instrumentation is unavailable on TT; stages still run on TTNN.

    def close(self):
        self.tt.close_device(self.device)
