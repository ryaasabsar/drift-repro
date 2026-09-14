"""Single-MI210 Qwen2.5-7B inference and Llama-Guard-3 evaluation using ROCm."""
from .common import ROOT
from .preset import Preset, main as run_preset

SUITE = ROOT / 'suites/mi210-qwen25-7b-llamaguard3.json'


def check_hardware():
    import torch
    if not getattr(torch.version, 'hip', None):
        raise RuntimeError('MI210 requires ROCm PyTorch and vLLM in .venv-rocm-vllm; CUDA wheels cannot run on AMD.')
    if not torch.cuda.is_available():
        raise RuntimeError('No usable AMD GPU. Check ROCm, device permissions, and HIP/ROCR_VISIBLE_DEVICES.')
    gpu = torch.cuda.get_device_properties(0)
    if 'MI210' not in gpu.name.upper() or gpu.total_memory < 55 * 1024**3:
        raise RuntimeError('This preset requires an Instinct MI210 with at least 55 GiB visible VRAM (64 GB board).')
    return {'gpu': gpu.name, 'memory_gib': round(gpu.total_memory / 1024**3, 1),
            'rocm_runtime': torch.version.hip}


PRESET = Preset('MI210', SUITE, check_hardware, 'HIP_VISIBLE_DEVICES')


def main(argv=None):
    run_preset(PRESET, argv)


if __name__ == '__main__':
    main()
