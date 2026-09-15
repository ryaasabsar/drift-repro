"""Checksummed result bundles: data only, no environments, weights, credentials or logs."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import zipfile
import uuid

from .common import WORKLOADS, now, read_json, write_json
from .evaluation import DATASET_DIR, EVALUATOR_VERSION
from .stages import refresh_status, result_lock, setting_directories, stage_status
from .tables import export_run

# Only these known artifacts are portable. Tables are regenerated on import/finalization.
SETTING_FILES = {'manifest.json', 'config.json', 'server.json', 'evaluation.json', 'summary.csv',
                 'code-results.jsonl', 'safety-labels.jsonl', 'stages.json'} | {f'{w}.jsonl' for w in WORKLOADS}
MAX_BYTES = 8 * 1024**3
MAX_FILES = 10000


def handoff(directory, target, output_dir='results/transfers'):
    """One file per transfer, retaining the existing checked tar format inside."""
    if target not in ('safety', 'evaluate'):
        raise ValueError('Handoff target must be safety or evaluate')
    root = Path(directory).resolve()
    status = stage_status(root)
    if target == 'evaluate' and any(s['safety']['status'] == 'pending' for s in status['settings'].values()):
        raise ValueError('Complete safety labeling on A100 before handing off to RTX')
    destination = Path(output_dir) / f'{root.name}-to-{target}-{uuid.uuid4().hex[:8]}.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='driftbench-handoff-') as temporary:
        archive = Path(temporary) / 'results.tar.gz'
        pack(root, archive)
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_STORED) as bundle:
            bundle.write(archive, archive.name)
            bundle.write(str(archive) + '.sha256', archive.name + '.sha256')
    return {'bundle': str(destination), 'bytes': destination.stat().st_size,
            'target_stage': target, 'next': f'Copy this ONE file to the next host, then: bash scripts/results.sh receive {destination}'}


def receive(bundle, output=None):
    bundle = Path(bundle).absolute()
    output = Path(output) if output else Path('results/runs') / bundle.stem
    with zipfile.ZipFile(bundle) as container, tempfile.TemporaryDirectory(prefix='driftbench-receive-') as temporary:
        names = ['results.tar.gz', 'results.tar.gz.sha256']
        if sorted(container.namelist()) != sorted(names):
            raise ValueError('Transfer bundle must contain exactly one archive and its checksum')
        for name in names:
            entry = container.getinfo(name)
            limit = MAX_BYTES if name.endswith('.gz') else 4096
            if entry.file_size > limit:
                raise ValueError('Transfer container exceeds size limits')
            with container.open(entry) as source, (Path(temporary) / name).open('xb') as target:
                shutil.copyfileobj(source, target)
        result = unpack(Path(temporary) / names[0], output)
    return {**result, 'next': f'bash scripts/results.sh stage status {output}'}


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Expected a regular result file: {source.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def pack(directory, output):
    root, output = Path(directory).resolve(), Path(output).absolute()
    checksum = Path(str(output) + '.sha256')
    if output.exists() or checksum.exists():
        raise ValueError('Archive/checksum already exists; choose a new stage or run name')
    if not output.name.endswith('.tar.gz'):
        raise ValueError('Use a .tar.gz archive filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    with result_lock(root) as paths, tempfile.TemporaryDirectory(prefix='driftbench-pack-') as temporary:
        status = stage_status(root)  # Refuse incomplete inference or stale evaluation artifacts.
        staged = Path(temporary) / 'data'
        staged.mkdir()
        if (root / 'suite.json').exists():
            copy_file(root / 'suite.json', staged / 'suite.json')
        for path in paths:
            destination = staged / path.relative_to(root)
            destination.mkdir(parents=True, exist_ok=True)
            for name in SETTING_FILES:
                if (path / name).exists():
                    copy_file(path / name, destination / name)
            manifest = read_json(path / 'manifest.json')
            dataset_dir = path / 'datasets' if (path / 'datasets').is_dir() else DATASET_DIR
            for workload in manifest['sources']:
                filename = WORKLOADS[workload][0]
                copy_file(dataset_dir / filename, destination / 'datasets' / filename)
        refresh_status(staged)
        files = {str(p.relative_to(staged).as_posix()): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
                 for p in sorted(staged.rglob('*')) if p.is_file()}
        if len(files) + 1 > MAX_FILES or sum(meta['bytes'] for meta in files.values()) > MAX_BYTES - 8 * 1024**2:
            raise ValueError('Result bundle exceeds the supported import size limits')
        index = {'schema_version': 1, 'run_id': root.name, 'created_at': now(),
                 'evaluator_version': EVALUATOR_VERSION, 'stages': status, 'files': files}
        write_json(staged / 'bundle.json', index)
        archive = Path(temporary) / 'result.tar.gz'
        with tarfile.open(archive, 'w:gz') as handle:
            for name in [*files, 'bundle.json']:
                handle.add(staged / name, arcname=name, recursive=False)
        # Exclusive destination creation avoids overwriting another export.
        with output.open('xb') as target, archive.open('rb') as source:
            shutil.copyfileobj(source, target)
        with checksum.open('x') as handle:
            handle.write(f'{file_hash(output)}  {output.name}\n')
    return {'archive': str(output), 'checksum': str(checksum), 'bytes': output.stat().st_size,
            'next': 'Transfer both files; unpack to a new result directory on the next host.'}


def unpack(archive, output):
    archive, output = Path(archive).absolute(), Path(output).absolute()
    checksum = Path(str(archive) + '.sha256')
    if output.exists() or output.is_symlink():
        raise ValueError('Destination already exists; unpack into a new directory')
    if not checksum.is_file():
        raise ValueError('Missing .sha256 sidecar; transfer it alongside the archive')
    expected = checksum.read_text().strip().split()
    if len(expected) != 2 or expected[1] != archive.name or expected[0] != file_hash(archive):
        raise ValueError('Archive checksum mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.driftbench-import-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'data'
        staged.mkdir()
        with tarfile.open(archive, 'r:gz') as handle:
            members, size = {}, 0
            for member in handle:
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or '..' in name.parts or
                        '\\' in member.name or str(name) != member.name or member.name in members):
                    raise ValueError('Archive contains unsafe or duplicate paths')
                size += member.size
                if len(members) >= MAX_FILES or size > MAX_BYTES:
                    raise ValueError('Result archive exceeds import size limits')
                members[member.name] = member
            if 'bundle.json' not in members or members['bundle.json'].size > 8 * 1024**2:
                raise ValueError('Missing/oversized bundle manifest')
            index = json.load(handle.extractfile(members['bundle.json']))
            if index.get('schema_version') != 1 or index.get('evaluator_version') != EVALUATOR_VERSION:
                raise ValueError('Unsupported bundle/evaluator version; use the matching runner checkout')
            files = index['files']
            if set(files) != set(members) - {'bundle.json'}:
                raise ValueError('Bundle file inventory mismatch')
            # Never extractall: write only validated regular files under a new private directory.
            for name, meta in files.items():
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if members[name].size != meta['bytes']:
                    raise ValueError(f'Bundle size mismatch: {name}')
                with handle.extractfile(members[name]) as source, destination.open('xb') as target:
                    shutil.copyfileobj(source, target)
                if file_hash(destination) != meta['sha256']:
                    raise ValueError(f'Bundle checksum mismatch: {name}')
        # Bundled datasets allow this validation without a local vendor checkout or downloads.
        report = refresh_status(staged)
        for path in setting_directories(staged):
            export_run(path)
        write_json(staged / 'transfer.json', {'schema_version': 1, 'imported_at': now(),
                                            'archive_sha256': expected[0], 'source_run_id': index['run_id']})
        if output.exists():
            raise ValueError('Destination appeared during import')
        os.rename(staged, output)
    return {'directory': str(output), **report,
            'next': f'Run stage status, safety, code, or final on {output}.'}
