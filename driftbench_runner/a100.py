"""Single-A100 Qwen2.5-7B inference followed by Llama-Guard-3 evaluation."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

from .common import ROOT, WORKLOADS, local_environment, read_json
from .logging import activity, configure, event
from .suite import load_plan

SUITE = ROOT / 'suites/a100-qwen25-7b-llamaguard3.json'
CREDENTIALS = ROOT / '.env'


def ensure_credentials_file(path=CREDENTIALS):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path
    with os.fdopen(descriptor, 'w') as handle:
        handle.write('HF_TOKEN=\n')
    return path


def load_credentials(path=CREDENTIALS):
    """Read data, never source shell code or include secret values in errors."""
    path = ensure_credentials_file(path)
    try:
        data = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].lstrip()
            key, separator, value = line.partition('=')
            key = key.strip()
            if not separator or key != 'HF_TOKEN' or key in data:
                raise ValueError
            values = shlex.split(value, comments=True, posix=True)
            if len(values) > 1:
                raise ValueError
            data[key] = values[0] if values else ''
    except (OSError, ValueError):
        raise ValueError('Cannot read credentials: .env must contain an HF_TOKEN assignment') from None
    token = data.get('HF_TOKEN', '').strip()
    if token:
        os.environ['HF_TOKEN'] = token


def check_hardware():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No usable NVIDIA GPU. Check the driver and CUDA_VISIBLE_DEVICES.')
    gpu = torch.cuda.get_device_properties(0)
    if 'A100' not in gpu.name.upper() or gpu.total_memory < 35 * 1024**3:
        raise RuntimeError('This preset requires an A100 with at least 35 GiB visible VRAM (40/80 GB board).')
    return {'gpu': gpu.name, 'memory_gib': round(gpu.total_memory / 1024**3, 1),
            'cuda_runtime': torch.version.cuda}


def check_model_access(model, revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    # Small files exercise gated repository access without fetching model weights.
    try:
        for filename in ('config.json', 'model.safetensors.index.json'):
            hf_hub_download(model, filename, revision=revision)
    except HfHubHTTPError:
        raise RuntimeError(f'Cannot access {model} at its pinned revision. Check repository approval, '
                           'Hugging Face authentication, and network access.') from None
    except Exception:
        raise RuntimeError(f'Cannot retrieve the pinned files for {model}; check network access or the local cache.') from None


def check_requirements(plan, inference_only):
    item = plan['settings'][0]
    config = item['config']
    with activity('Checking A100 hardware', stage='preflight'):
        hardware = check_hardware()
    event('A100 detected', stage='preflight', **hardware)
    if not inference_only and 'safety' in plan['workloads']:
        ev = plan['evaluation']
        with activity('Checking LlamaGuard access', stage='preflight', model=ev['judge_model']):
            check_model_access(ev['judge_model'], ev['judge_revision'])
    with activity('Checking Qwen model access', stage='preflight', model=config['model']):
        check_model_access(config['model'], config['revision'])
    if not inference_only and 'code' in plan['workloads']:
        from .evaluation import execute_code
        with activity('Checking isolated code evaluation', stage='preflight'):
            result = execute_code({'prompt': 'def check_value():\n', 'entry_point': 'check_value',
                                   'test_cases': 'def check(candidate):\n    assert candidate() == 1'}, '    return 1\n')
            if not result.get('correct'):
                raise RuntimeError('The isolated HumanEval worker is unavailable. Install/enable bubblewrap, '
                                   'or use --inference-only and score later on the evaluation host.')
    from .inference import prepare
    with activity('Checking tokenizer and selected prompt lengths', stage='preflight', model=config['model']):
        profile = SUITE.parent / read_json(SUITE)['settings'][0]['config']
        _, _, rows, _ = prepare(profile, plan['workloads'], plan['limit'])
        required = max(len(row['input_ids']) for row in rows) + config['generation']['max_tokens']
        if required > config['engine']['max_model_len']:
            raise RuntimeError(f'Prompt budget {required} exceeds the configured context; no prompt will be truncated.')
    return {**hardware, 'selected_prompts': len(rows), 'required_context': required,
            'model': config['model'], 'revision': config['revision'],
            'evaluation': None if inference_only else plan['evaluation']}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', default='results/a100-qwen25-7b-llamaguard3')
    result.add_argument('--limit', type=int, help='First N prompts per workload; omit for all 2,284')
    result.add_argument('--workloads', nargs='+', choices=list(WORKLOADS), default=list(WORKLOADS))
    result.add_argument('--device', help='One GPU index; omitted preserves CUDA_VISIBLE_DEVICES from the scheduler')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--inference-only', action='store_true', help='Defer all evaluation, including gated LlamaGuard')
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Show the plan without credentials, downloads, or GPU access')
    mode.add_argument('--check', action='store_true', help='Check hardware, model access, tokenizer and evaluator; do not run inference')
    result.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], default='info')
    result.add_argument('--progress-interval', type=float, default=15)
    return result


def suite_arguments(args):
    command = ['-m', 'driftbench_runner', 'suite', '--config', str(SUITE), '--output', args.output,
               '--workloads', *args.workloads, '--log-level', args.log_level,
               '--progress-interval', str(args.progress_interval)]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    if args.device is not None:
        command += ['--device', args.device]
    for name in ('dry_run', 'resume', 'inference_only'):
        if getattr(args, name):
            command.append('--' + name.replace('_', '-'))
    return command


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    if args.device is not None and not re.fullmatch(r'\d+', args.device):
        options.error('--device must name one GPU index')
    try:
        configure(args.log_level, interval=args.progress_interval)
        plan = load_plan(SUITE, args.output, args.limit, args.workloads, args.device)
        if not args.dry_run:
            local_environment()
            load_credentials()
            if args.device is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            checked = check_requirements(plan, args.inference_only)
            if args.check:
                print(json.dumps(checked, indent=2))
                return
    except KeyboardInterrupt:
        event('A100 preflight interrupted', stage='preflight', level='warning')
        raise SystemExit(130) from None
    except Exception as exc:
        # No raw Hub response, environment dump or credential contents in logs.
        event(str(exc), stage='preflight', level='error')
        raise SystemExit(1) from None
    os.execv(sys.executable, [sys.executable, *suite_arguments(args)])


if __name__ == '__main__':
    main()
