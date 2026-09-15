"""Repeat a response across server restarts; tiny profiles are not study results."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from driftbench_runner.common import read_json, read_jsonl, write_json

work = ROOT / '.archive/check-output'
work.mkdir(parents=True, exist_ok=True)
settings = []
for backend in ('vllm', 'sglang'):
    production = read_json(ROOT / 'configs' / f'a100-qwen35-9b-base-{backend}-http.json')
    config = read_json(ROOT / 'configs' / f'a100-qwen35-9b-base-{backend}-http.json')
    config.update(model='Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17',
                  hardware={'vendor': 'nvidia', 'device': 'RTX 3060 Laptop 6GB'},
                  prompt_format='chat', chat_template_kwargs={'enable_thinking': False})
    config['generation'] = production['generation'].copy()
    config['seed'] = production['seed']
    config['launch']['env'] = production['launch']['env'].copy()
    config['setup_id'] = f'rtx_refactor_{backend}'
    config['batch_size'] = 1
    config['engine']['max_model_len'] = 2048
    config['generation']['max_tokens'] = 32
    if backend == 'vllm':
        config['engine'].update(max_num_seqs=1, gpu_memory_utilization=0.5)
    else:
        config['engine'].update(max_running_requests=1, mem_fraction_static=0.5,
                                max_mamba_cache_size=1, disable_piecewise_cuda_graph=True)
        for key in ('enable_deterministic_inference', 'sampling_defaults'):
            config['engine'][key] = production['engine'][key]
    config['server']['base_url'] = f'http://127.0.0.1:{18001 if backend == "vllm" else 18002}'
    path = work / f'{backend}.json'
    write_json(path, config)
    for repeat in (1, 2):
        settings.append({'id': f'{config["setup_id"]}_{repeat}', 'config': str(path),
                         'server_python': str(ROOT / ('.venv' if backend == 'vllm' else '.venv-sglang') / 'bin/python')})
suite = work / 'rtx-smoke.json'
write_json(suite, {'suite_id': 'rtx-refactor-smoke', 'workloads': ['math'], 'limit': 1, 'settings': settings})
run_id = 'rtx-smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
subprocess.run([sys.executable, '-m', 'driftbench_runner', 'infer', '--config', str(suite),
                '--run-id', run_id, '--output-root', str(work)], cwd=ROOT,
               env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'PYTHONHASHSEED': '42', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}, check=True)
report = {}
for backend in ('vllm', 'sglang'):
    responses = [read_jsonl(work / run_id / 'settings' / f'rtx_refactor_{backend}_{repeat}' / 'math.jsonl')[0]
                 for repeat in (1, 2)]
    same = all(responses[0][key] == responses[1][key]
               for key in ('output_text', 'output_token_ids', 'finish_reason'))
    # Measure residual drift rather than treating greedy decoding as a guarantee.
    assert all(r['output_token_ids'] for r in responses)
    for key in ('input_token_ids', 'effective_sampling'):
        assert responses[0][key] == responses[1][key], f'{backend}: controls changed'
    report[backend] = {'matching_repeats': same, 'responses': 2,
                       'output_tokens': responses[0]['output_tokens']}
write_json(work / run_id / 'repeatability.json', report)
print('Saved smoke run:', work / run_id)
