"""Exercise filtered inference, two portable hops, scoring, comparisons and diagnostics."""
import json
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from driftbench_runner.common import ROOT, digest, read_json, write_json
from driftbench_runner import comparison, hardware, software, stages, suite, transfer
from driftbench_runner.compare_many import catalog, compare_many
from driftbench_runner.serving import serving_identity
from test_stages import make_run, fake_workers


@pytest.mark.parametrize('host,framework,count', [('a100','vllm',3), ('a100','sglang',3),
    ('mi210','vllm',3), ('mi210','sglang',3), ('blackhole-p150b','vllm',3)])
def test_framework_selection_contains_only_requested_models(host, framework, count, tmp_path):
    plan = suite.run_suite(ROOT / 'suites' / f'{host}.json', tmp_path / 'run',
                           frameworks=[framework], dry_run=True)
    assert len(plan['settings']) == count
    assert len({s['config']['model'] for s in plan['settings']}) == 3
    assert {s['config']['backend'] for s in plan['settings']} == {framework}
    assert set(plan['selected']) == {s['id'] for s in plan['settings']}
    assert not (tmp_path / 'run').exists()


def test_invalid_or_conflicting_selection_fails(tmp_path):
    with pytest.raises(ValueError, match='Frameworks unavailable'):
        suite.load_plan(ROOT / 'suites/blackhole-p150b.json', tmp_path, frameworks=['sglang'])
    with pytest.raises(ValueError, match='Unknown settings'):
        suite.load_plan(ROOT / 'suites/a100.json', tmp_path, selected=['typo'])


def test_filtered_run_survives_both_hops_and_arbitrary_comparisons(tmp_path, monkeypatch, fake_workers):
    # This mocked integration has no accelerator; do not contend with a real smoke run.
    monkeypatch.setattr(suite, 'ROOT', tmp_path)
    def inference(item, plan, resume, environment):
        fixture = make_run(tmp_path / ('fixture-' + item['id']), plan['workloads'], plan['limit'])
        manifest = read_json(fixture / 'manifest.json')
        manifest['config'] = item['config']
        server = manifest['server_provenance']
        server['identity'] = serving_identity(item['config'])
        server['serving_fingerprint'] = digest(server['identity'])
        _, outputs = stages.load_run(fixture)
        identity = {'config': item['config'], 'sources': manifest['sources'], 'server': server['serving_fingerprint'],
                    'requests': [[*k,r['request_sha256'],r['source_sha256']] for k,r in outputs.items()]}
        # Reconstruct ordering from the benchmark, as the real runner does.
        from driftbench_runner.common import load_workloads
        rows, _ = load_workloads(plan['workloads'], plan['limit'])
        identity['requests'] = [[r['workload'],r['prompt_id'], outputs[(r['workload'],r['prompt_id'])]['request_sha256'],r['source_sha256']] for r in rows]
        manifest['run_fingerprint'] = digest(identity)
        write_json(fixture / 'manifest.json', manifest)
        shutil.copytree(fixture, item['run_dir'], dirs_exist_ok=True)
    monkeypatch.setattr(suite, 'run_inference', inference)
    origin = tmp_path / 'origin'
    state = suite.run_suite(ROOT / 'suites/a100.json', origin, frameworks=['vllm'], limit=1)
    assert state['status'] == 'inference_complete' and len(state['settings']) == 3
    assert suite.run_suite(ROOT / 'suites/a100.json', origin, frameworks=['vllm'], limit=1,
                           resume=True)['status'] == 'inference_complete'
    with pytest.raises(ValueError, match='changed'):
        suite.run_suite(ROOT / 'suites/a100.json', origin, frameworks=['sglang'], limit=1, resume=True)
    # It is a complete three-setting run, not a six-setting run with pending entries.
    first = transfer.handoff(origin, 'safety', tmp_path / 'transfers')
    a100 = tmp_path / 'a100-renamed'
    transfer.receive(first['bundle'], a100)
    stages.run_stage(a100, 'safety')
    second = transfer.handoff(a100, 'evaluate', tmp_path / 'transfers')
    rtx = tmp_path / 'rtx-renamed'
    transfer.receive(second['bundle'], rtx)
    assert stages.run_stage(rtx, 'evaluate')['status'] == 'complete'
    paths = stages.setting_directories(rtx)
    original_paths = stages.setting_directories(origin)
    assert all((a/'manifest.json').read_bytes() == (b/'manifest.json').read_bytes() for a,b in zip(paths,original_paths))
    assert len(catalog([rtx, rtx])) == 3
    result = compare_many(paths[1], [rtx], tmp_path / 'comparisons', allow_confounded=True)
    assert result['status'] == 'complete' and len(result['comparisons']) == 2
    assert (tmp_path / 'comparisons/comparisons.csv').is_file()
    report = read_json(result['comparisons'][0]['report'])
    assert report['workloads']['math']['paired'] == 1
    assert report['workloads']['safety']['scored_pairs'] == 1
    assert report['workloads']['code']['scored_pairs'] == 1
    assert report['workloads']['math']['token_comparable_pairs'] == 0  # Different models.


def test_single_file_receive_checks_inner_checksum(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    bundle = transfer.handoff(run, 'safety', tmp_path)['bundle']
    with zipfile.ZipFile(bundle) as z:
        data = {n:z.read(n) for n in z.namelist()}
    data['results.tar.gz'] += b'corrupt'
    with zipfile.ZipFile(bundle, 'w') as z:
        for n,content in data.items():
            z.writestr(n, content)
    with pytest.raises(ValueError, match='checksum mismatch'):
        transfer.receive(bundle, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_incompatible_comparison_is_not_reported_as_zero_drift(tmp_path):
    a=make_run(tmp_path/'a',['math'],limit=1)
    b=make_run(tmp_path/'b',['math'],limit=2)
    c=make_run(tmp_path/'c',['math'],limit=1)
    report=compare_many(a,[b,c],tmp_path/'comparisons',allow_confounded=True)
    assert report['status']=='incompatible'
    assert [r['status'] for r in report['comparisons']]==['incompatible','complete']
    assert 'identical prompt-ID sets' in report['comparisons'][0]['error']


def test_evaluate_and_handoff_require_safety_first(tmp_path, fake_workers):
    run = make_run(tmp_path / 'run')
    with pytest.raises(ValueError, match='safety'):
        stages.run_stage(run, 'evaluate')
    assert not fake_workers
    with pytest.raises(ValueError, match='safety'):
        transfer.handoff(run, 'evaluate', tmp_path)


def test_client_package_difference_is_visible_and_strict_comparison_refuses(tmp_path):
    a,b = [make_run(tmp_path / name, ['math']) for name in ('a','b')]
    for path,version in ((a,'4.57.6'), (b,'5.3.0')):
        meta = read_json(path / 'manifest.json')
        meta['client_environment'] = {'python':'3.12.14', 'packages': {'transformers':version}}
        write_json(path / 'manifest.json', meta)
    with pytest.raises(ValueError, match='client_software'):
        comparison.compare_runs(a,b,tmp_path/'strict')
    report = comparison.compare_runs(a,b,tmp_path/'explicit',allow_confounded=True)
    assert 'client_software' in report['confounds']
    assert report['baseline_client']['packages']['transformers'] == '4.57.6'


def test_runtime_contract_rejects_wrong_cuda_and_records_vendor_build():
    config = read_json(ROOT / 'configs/a100-qwen25-7b-vllm-http.json')
    env = {'python':'3.12.14','cuda_runtime':'13.0', 'packages':{
        'vllm':'0.17.1','torch':'2.10.0+cu130','transformers':'4.57.6','tokenizers':'0.22.2'}}
    assert software.check_runtime(config,env)['status'] == 'mismatch'
    env['cuda_runtime']='12.8'
    env['packages']['torch']='2.10.0+cu128'
    assert software.check_runtime(config,env)['status'] == 'match'


def test_numerical_environment_difference_is_reported(tmp_path):
    a,b = [make_run(tmp_path / name, ['math']) for name in ('a','b')]
    meta = read_json(b / 'manifest.json')
    meta['environment']['runtime_environment'] = {'VLLM_BATCH_INVARIANT': '1'}
    write_json(b / 'manifest.json',meta)
    report = comparison.compare_runs(a,b,tmp_path/'comparison',allow_confounded=True)
    assert 'numerical_environment' in report['confounds']


def test_cuda_probe_retries_fresh_process_and_reports_timeouts(monkeypatch):
    calls = []
    def launch(argv, **kwargs):
        calls.append((argv,kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(
            {'accelerator_error':'initialization failed'} if len(calls)==1 else {'accelerators':[{'name':'A100'}]}))
    monkeypatch.setattr(hardware.subprocess, 'run', launch)
    assert hardware.probe_accelerators()['accelerators'][0]['name'] == 'A100'
    assert len(calls)==2 and all(c[1]['timeout']==30 for c in calls)
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired(args[0],kwargs['timeout'])
    monkeypatch.setattr(hardware.subprocess,'run',timeout)
    assert 'timed out' in hardware.probe_accelerators()['accelerator_error']


def test_cuda_diagnostics_identify_mask_stubs_and_driver_mismatch():
    report = {'runtime_environment':{'CUDA_VISIBLE_DEVICES':'-1','LD_LIBRARY_PATH':'/cuda/lib64/stubs'},
              'cuda_runtime':'13.0','nvidia':{'stdout':'A100, GPU-id, 40000, 570.195.03, 8.0'}}
    advice = ' '.join(hardware.hardware_guidance(report))
    assert 'CUDA 13' in advice and 'hides all GPUs' in advice and 'stub libraries' in advice
