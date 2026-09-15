"""Run CUDA/HIP initialization in a fresh, bounded child process."""
import json


def probe():
    result = {'accelerators': [], 'cuda_runtime': None, 'rocm_runtime': None}
    try:
        import torch
        result.update(cuda_runtime=torch.version.cuda, rocm_runtime=getattr(torch.version, 'hip', None))
        if not (result['cuda_runtime'] or result['rocm_runtime']):
            return result
        torch.cuda.init()
        for index in range(torch.cuda.device_count()):
            value = torch.ones(1, device=f'cuda:{index}')
            if (value + 1).item() != 2:
                raise RuntimeError('Accelerator calculation failed')
            properties = torch.cuda.get_device_properties(index)
            result['accelerators'].append({'name': properties.name,
                'capability': list(torch.cuda.get_device_capability(index)),
                'memory_bytes': properties.total_memory})
    except Exception as exc:
        result.update(accelerators=[], accelerator_error=f'{type(exc).__name__}: {exc}')
    return result


if __name__ == '__main__':
    print(json.dumps(probe()))
