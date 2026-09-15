import io
import json
import tarfile

import pytest

from driftbench_runner import transfer
from test_stages import make_run


def rewrite_archive(path, transform):
    with tarfile.open(path, 'r:gz') as handle:
        files = {m.name: handle.extractfile(m).read() for m in handle.getmembers()}
    files = transform(files)
    with tarfile.open(path, 'w:gz') as handle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
    path.with_name(path.name + '.sha256').write_text(f'{transfer.file_hash(path)}  {path.name}\n')


def test_pack_allowlist_excludes_credentials_logs_weights_and_locks(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    for name in ('.env', 'token.txt', 'model.safetensors', '.run.lock', 'unrelated.json'):
        (run / name).write_text('private fixture')
    (run / 'logs').mkdir()
    (run / 'logs' / 'server.log').write_text('private log fixture')
    archive = tmp_path / 'run.tar.gz'
    transfer.pack(run, archive)
    with tarfile.open(archive, 'r:gz') as handle:
        names = handle.getnames()
        assert 'datasets/gsm8k_prompts.jsonl' in names
        assert not any('private fixture' in handle.extractfile(m).read().decode() for m in handle.getmembers())
        assert all(not n.startswith('logs/') for n in names)
    with pytest.raises(ValueError, match='already exists'):
        transfer.pack(run, archive)


def test_unpack_requires_sidecar_and_never_overwrites(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    archive = tmp_path / 'run.tar.gz'
    transfer.pack(run, archive)
    destination = tmp_path / 'import'
    destination.mkdir()
    sentinel = destination / 'keep'
    sentinel.write_text('existing run')
    with pytest.raises(ValueError, match='already exists'):
        transfer.unpack(archive, destination)
    assert sentinel.read_text() == 'existing run'
    archive.with_name(archive.name + '.sha256').unlink()
    with pytest.raises(ValueError, match='Missing .sha256'):
        transfer.unpack(archive, tmp_path / 'new')


def test_outer_and_internal_checksums_verified_without_partial_import(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    archive = tmp_path / 'run.tar.gz'
    transfer.pack(run, archive)
    original = archive.read_bytes()
    archive.write_bytes(original + b'corrupted')
    with pytest.raises(ValueError, match='Archive checksum mismatch'):
        transfer.unpack(archive, tmp_path / 'new')
    archive.write_bytes(original)
    def change(files):
        files['math.jsonl'] = files['math.jsonl'].replace(b'fixture output', b'changed output')
        return files
    rewrite_archive(archive, change)
    with pytest.raises(ValueError, match='Bundle (size|checksum) mismatch'):
        transfer.unpack(archive, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()
    assert not list(tmp_path.glob('.driftbench-import-*'))


@pytest.mark.parametrize('kind,name', [('file', '../escape'), ('file', '/absolute'),
                                       ('symlink', 'linked'), ('hardlink', 'linked'), ('file', './ambiguous')])
def test_unsafe_archive_paths_and_links_rejected(tmp_path, kind, name):
    archive = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        member = tarfile.TarInfo(name)
        if kind == 'symlink':
            member.type = tarfile.SYMTYPE
            member.linkname = '/tmp'
        elif kind == 'hardlink':
            member.type = tarfile.LNKTYPE
            member.linkname = '../outside'
        handle.addfile(member)
    archive.with_name(archive.name + '.sha256').write_text(f'{transfer.file_hash(archive)}  {archive.name}\n')
    with pytest.raises(ValueError, match='unsafe'):
        transfer.unpack(archive, tmp_path / 'new')
    assert not (tmp_path / 'new').exists()


def test_archive_inventory_cannot_hide_extra_files(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    archive = tmp_path / 'run.tar.gz'
    transfer.pack(run, archive)
    rewrite_archive(archive, lambda files: {**files, 'unexpected': b'extra'})
    with pytest.raises(ValueError, match='inventory mismatch'):
        transfer.unpack(archive, tmp_path / 'new')


def test_bundle_version_mismatch_rejected(tmp_path):
    run = make_run(tmp_path / 'run', ['math'])
    archive = tmp_path / 'run.tar.gz'
    transfer.pack(run, archive)
    def change(files):
        index = json.loads(files['bundle.json'])
        index['evaluator_version'] = 'future'
        files['bundle.json'] = json.dumps(index).encode()
        return files
    rewrite_archive(archive, change)
    with pytest.raises(ValueError, match='version'):
        transfer.unpack(archive, tmp_path / 'new')


def test_pack_rejects_bundle_larger_than_receiver_limit(tmp_path, monkeypatch):
    run = make_run(tmp_path / 'run', ['math'])
    monkeypatch.setattr(transfer, 'MAX_BYTES', 8 * 1024**2)
    with pytest.raises(ValueError, match='size limits'):
        transfer.pack(run, tmp_path / 'too-large.tar.gz')
    assert not (tmp_path / 'too-large.tar.gz').exists()
