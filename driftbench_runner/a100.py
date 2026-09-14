"""Single-A100 Qwen2.5-7B inference followed by Llama-Guard-3 evaluation."""
from .common import ROOT
from .preset import Preset, main as run_preset

SUITE = ROOT / 'suites/a100-qwen25-7b-llamaguard3.json'


def check_hardware():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No usable NVIDIA GPU. Check the driver and CUDA_VISIBLE_DEVICES.')
    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name.upper() or gpu.total_memory < 35 * 1024**3:
        raise RuntimeError('This preset requires an A100 with at least 35 GiB visible VRAM (40/80 GB board).')
    return {'gpu': gpu.name, 'memory_gib': round(gpu.total_memory / 1024**3, 1),
            'cuda_runtime': torch.version.cuda}



PRESET = Preset('A100', SUITE, check_hardware, 'CUDA_VISIBLE_DEVICES')


def main(argv=None):
    run_preset(PRESET, argv)


if __name__ == '__main__':
    main()
