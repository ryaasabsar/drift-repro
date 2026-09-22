"""Print the three recorded divergence cases using the original full-shape kernels.

Run with your CUDA/ROCm Python: python normbench/reproduce_examples.py
No model, previous results, or credentials are needed. This is one diagnostic
pass; use `run.sh intervene` for the full repetition and attribution controls.
"""
import numpy as np
import torch
import triton

from normbench.backends import TorchBackend
from normbench.data import logical_bits
from normbench.intervention_kernel import launch
from normbench.interventions import load_bundle


def bits(tensor):
    return logical_bits(tensor.float().cpu().numpy(), str(tensor.dtype).removeprefix('torch.'))


def value(tensor, row, column):
    return tensor[row, 0 if tensor.shape[1] == 1 else column].item()


def main():
    platform = 'amd' if torch.version.hip else 'nvidia'
    backend = TorchBackend(platform)
    manifest, data = load_bundle()
    print(f"GPU: {backend.info['device']}")
    print(f"PyTorch: {torch.__version__}; Triton: {triton.__version__}")
    print(f"CUDA: {torch.version.cuda}; HIP: {torch.version.hip}")

    # Keep all 1552 x 128 inputs: slicing changes the kernel's execution layout.
    inputs = {name: backend.upload(data[name], 'bfloat16') for name in ('x', 'w', 'z')}
    shared = {name: backend.upload(data[f'shared_{name}'], 'float32')
              for name in ('rsqrt', 'sigmoid')}
    eps = manifest['case']['eps']

    def run(replacements, flags=None, capture=True):
        switches = None if flags is None else torch.tensor(flags, dtype=torch.int32, device='cuda')
        return launch(inputs, replacements, eps, capture, switches)[0]

    # y = BF16((x * rsqrt(mean(x*x) + eps) * w) * (z * sigmoid(z)))
    original = run(shared)
    native = run(shared, capture=False)
    replay = run(shared, (0, 0))
    shared_sigmoid = run(shared, (0, 1))
    own = {name: replay[name] for name in ('rsqrt', 'sigmoid')}
    self_rsqrt = run(own, (1, 0))

    print('Original matches historical stage values:', all(
        np.array_equal(bits(t), data[f'{platform}/trace/{s}']) for s, t in original.items()))
    print('Original captured/native output agrees:',
          np.array_equal(bits(original['output']), bits(native['output'])))
    print('Diagnostic replay matches original stages:', all(
        np.array_equal(bits(original[s]), bits(replay[s])) for s in original))

    for row, column in ((761, 11), (1330, 36), (1385, 8)):
        print(f'\nPosition ({row}, {column}), zero-based')
        print(f"x={float(data['x'][row, column])!r}, w={float(data['w'][0, column])!r}, "
              f"z={float(data['z'][row, column])!r}")
        for stage in ('add_epsilon', 'rsqrt', 'sigmoid', 'normalized', 'weighted', 'gate', 'precast', 'output'):
            print(f'  {stage:14s} {value(original[stage], row, column)!r}')
        print('  shared sigmoid output:', value(shared_sigmoid['output'], row, column))
        print('  full-expression FP64 reference:', float(data['reference/output'][row, column]))

    print('\nSelf-rsqrt replay at (5, 0): normalized = x * rsqrt')
    print('  x:', float(data['x'][5, 0]))
    for name, trace in (('computed', replay), ('reloaded own value', self_rsqrt)):
        print(f"  {name}: rsqrt={value(trace['rsqrt'], 5, 0)!r}, "
              f"normalized={value(trace['normalized'], 5, 0)!r}, "
              f"BF16 output={value(trace['output'], 5, 0)!r}")
    print('  rsqrt bits unchanged:', np.array_equal(bits(replay['rsqrt']), bits(self_rsqrt['rsqrt'])))
    print('  changed normalized elements:', np.count_nonzero(
        bits(replay['normalized']) != bits(self_rsqrt['normalized'])))
    print('\nOne diagnostic pass; use run.sh intervene for full repeat/control checks.')
    backend.close()


if __name__ == '__main__':
    main()
