"""Experimental Blackhole P150b Qwen2.5-7B preset; Llama Guard runs on the host CPU."""
from .common import ROOT
from .hardware import discover, require_hardware
from .preset import Preset, main as run_preset

SUITE = ROOT / 'suites/blackhole-p150b-qwen25-7b-llamaguard3.json'


def check_hardware():
    hardware = discover()
    require_hardware('tenstorrent', hardware)
    if not hardware['tenstorrent_device_nodes']:
        raise RuntimeError('Tenstorrent PCI device found, but /dev/tenstorrent is unavailable; check driver/device mounts.')
    # tt-kmd enumerate.h identifies Blackhole as 1e52:b140; this does not identify the board SKU.
    if not any(device['vendor_id'] == '0x1e52' and device['device_id'] == '0xb140'
               for device in hardware['pci_devices']):
        raise RuntimeError('No Blackhole PCI device found; the P150b preset cannot run on Wormhole.')
    if 'vllm' not in hardware['packages'] or 'ttnn' not in hardware['packages']:
        raise RuntimeError('Use a TT-Metal/TTNN-enabled vLLM environment in .venv-tt-vllm.')
    # PCI presence does not establish the board variant or model kernel support.
    return {'requested_board': 'Blackhole P150b', 'board_verified': False,
            'tenstorrent_device_nodes': hardware['tenstorrent_device_nodes'],
            'pci_devices': hardware['pci_devices'], 'packages': hardware['packages']}


PRESET = Preset('Blackhole P150b', SUITE, check_hardware, None, experimental=True)


def main(argv=None):
    run_preset(PRESET, argv)


if __name__ == '__main__':
    main()
