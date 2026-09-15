"""Shared client pins and explicit vendor serving contracts."""
import importlib.metadata
import platform
import re

from .common import ROOT, digest, read_json


def client_pins():
    return dict(re.findall(r'^([a-zA-Z0-9_.-]+)==([^\s]+)$',
                           (ROOT / 'requirements.client.lock.txt').read_text(), re.MULTILINE))


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def client_environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': package_versions(client_pins()),
            'runner_sha256': digest({p.name: digest(p.read_bytes()) for p in sorted((ROOT / 'driftbench_runner').glob('*.py'))}),
            'requirements_sha256': digest((ROOT / 'requirements.client.lock.txt').read_bytes())}


def require_client():
    current = client_environment()
    expected = read_json(ROOT / 'runtime-contracts.json')['client']
    differences = {name: {'expected': version, 'actual': current['packages'].get(name)}
                   for name, version in client_pins().items() if current['packages'].get(name) != version}
    if current['python'] != expected['python']:
        differences['python'] = {'expected': expected['python'], 'actual': current['python']}
    if differences:
        raise ValueError(f'Common tokenizer client differs: {differences}. Run bash scripts/bootstrap.sh --client-only and use .venv-client/bin/python')
    return current


def check_runtime(config, metadata):
    key = config['hardware']['vendor'] + '/' + config['backend']
    contract = read_json(ROOT / 'runtime-contracts.json')['serving'][key]
    mismatches = {}
    for name, version in contract['packages'].items():
        actual = metadata['packages'].get(name)
        # Vendor build suffixes are retained in provenance, but share an upstream release.
        if actual is None or actual.split('+')[0] != version:
            mismatches[name] = {'expected_release': version, 'actual': actual}
    if not metadata['python'].startswith('3.12.'):
        mismatches['python'] = {'expected': '3.12.x', 'actual': metadata['python']}
    if 'cuda_runtime' in contract and metadata.get('cuda_runtime') != contract['cuda_runtime']:
        mismatches['cuda_runtime'] = {'expected': contract['cuda_runtime'], 'actual': metadata.get('cuda_runtime')}
    if key.startswith('amd/') and not metadata.get('rocm_runtime'):
        mismatches['rocm_runtime'] = {'expected': 'ROCm build', 'actual': metadata.get('rocm_runtime')}
    return {'profile': key, 'status': 'match' if not mismatches else 'mismatch',
            'differences': mismatches, 'note': contract['note']}
