from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(ROOT / "scripts/install_tt_vllm.sh", scripts)
    return tmp_path


def invoke(workspace, *args):
    return subprocess.run(["bash", str(workspace / "scripts/install_tt_vllm.sh"), *args],
                          cwd="/tmp", capture_output=True, text=True)


def test_dry_run_does_not_create_environment_or_sources(workspace):
    before = sorted(workspace.rglob("*"))
    result = invoke(workspace, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--python 3.12" in result.stdout
    assert sorted(workspace.rglob("*")) == before


@pytest.mark.parametrize("symlink", [False, True])
def test_recreate_refuses_non_environment_and_symlink(workspace, symlink):
    original = workspace / ".venv"
    original.mkdir()
    (original / "keep.txt").write_text("existing CUDA environment")
    target = workspace / ".venv-tt-vllm"
    if symlink:
        target.symlink_to(original, target_is_directory=True)
    else:
        target.mkdir()
    result = invoke(workspace, "--recreate")
    assert result.returncode != 0
    assert "Refusing" in result.stderr
    assert (original / "keep.txt").read_text() == "existing CUDA environment"
    assert not (workspace / ".tools").exists()


def test_existing_environment_is_not_cleared_without_explicit_flag(workspace):
    target = workspace / ".venv-tt-vllm"
    target.mkdir()
    (target / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.10.12\n")
    result = invoke(workspace, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--clear" not in result.stdout
    assert "Existing" in result.stdout
    replacement = invoke(workspace, "--dry-run", "--recreate")
    assert replacement.returncode == 0, replacement.stderr
    assert "--clear" in replacement.stdout
    assert "3.10.12" in (target / "pyvenv.cfg").read_text()


def test_failed_environment_creation_stops_before_source_or_package_install(workspace):
    tools = workspace / ".tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("#!/bin/sh\nexit 23\n")
    uv.chmod(0o755)
    (workspace / "scripts/ensure_uv.sh").write_text("#!/bin/sh\nexit 0\n")
    result = invoke(workspace)
    assert result.returncode == 23, result.stderr
    assert "git init" not in result.stdout
    assert "pip install" not in result.stdout
    assert not (workspace / ".venv-tt-vllm").exists()


def test_missing_check_environment_does_not_install(workspace):
    result = invoke(workspace, "--check")
    assert result.returncode != 0
    assert "run without --check" in result.stderr
    assert not (workspace / ".tools").exists()


def test_check_cannot_recreate_an_environment(workspace):
    result = invoke(workspace, "--check", "--recreate")
    assert result.returncode == 2
    assert not (workspace / ".tools").exists()
