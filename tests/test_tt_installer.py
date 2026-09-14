from pathlib import Path
import shutil
import subprocess
import sys

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


def fake_toolchain(workspace, compiler_exit=0):
    """Exercise repair orchestration without loading TTNN or requiring hardware."""
    target = workspace / ".venv-tt-vllm"
    (target / "bin").mkdir(parents=True)
    (target / "pyvenv.cfg").write_text("version = 3.12\n")
    # Use a wrapper: pyvenv.cfg must not change the test interpreter's stdlib.
    (target / "bin/python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (target / "bin/python").chmod(0o755)
    wheel = workspace / "ttnn"
    wheel.mkdir()
    (wheel / "__init__.py").write_text("raise AssertionError('Must not import TTNN')\n")
    dist = workspace / "ttnn-0.77.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: ttnn\nVersion: 0.77.0\n")
    metal = workspace / ".tools/tt-metal-9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9"
    metal.mkdir(parents=True)
    sfpi = workspace / ".tools/sfpi-7.69.0"
    (sfpi / "compiler/bin").mkdir(parents=True)
    compiler = sfpi / "compiler/bin/riscv-tt-elf-g++"
    compiler.write_text(f'#!/bin/sh\n[ "$1" = --version ] && exit 0\ncat >/dev/null\nexit {compiler_exit}\n')
    compiler.chmod(0o755)
    (sfpi / ".driftbench-archive-sha256").write_text(
        "b0c93362de2f69b0e4c335abd126ff59748f5dae316980d17aa1525ed523eb14\n"
    )
    return wheel, metal, sfpi


def test_toolchain_repair_registers_both_roots_without_installing_python(workspace):
    wheel, metal, sfpi = fake_toolchain(workspace)
    result = invoke(workspace, "--toolchain-only")
    assert result.returncode == 0, result.stderr
    assert "pip install" not in result.stdout
    for root in (wheel, metal):
        assert (root / "runtime/sfpi").resolve() == sfpi
    before = sorted(workspace.rglob("*"))
    check = invoke(workspace, "--check", "--toolchain-only")
    assert check.returncode == 0, check.stderr
    assert sorted(workspace.rglob("*")) == before


def test_failed_compiler_probe_does_not_register_runtime_links(workspace):
    wheel, metal, _ = fake_toolchain(workspace, compiler_exit=42)
    result = invoke(workspace, "--toolchain-only")
    assert result.returncode == 42, result.stderr
    assert not (wheel / "runtime").exists()
    assert not (metal / "runtime").exists()


def test_toolchain_check_does_not_repair_missing_links(workspace):
    wheel, metal, _ = fake_toolchain(workspace)
    result = invoke(workspace, "--check", "--toolchain-only")
    assert result.returncode != 0
    assert "run installer --toolchain-only" in result.stderr
    assert not (wheel / "runtime").exists()
    assert not (metal / "runtime").exists()


def test_toolchain_repair_preserves_existing_runtime_compiler(workspace):
    wheel, _, _ = fake_toolchain(workspace)
    existing = wheel / "runtime/sfpi"
    existing.mkdir(parents=True)
    (existing / "keep").write_text("custom compiler")
    result = invoke(workspace, "--toolchain-only")
    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert (existing / "keep").read_text() == "custom compiler"


def test_toolchain_download_rejects_bad_checksum_and_cleans_staging(workspace):
    wheel, _, sfpi = fake_toolchain(workspace)
    shutil.rmtree(sfpi)
    curl = workspace / ".tools/curl"
    curl.write_text('#!/bin/sh\nfor arg; do output="$arg"; done\nprintf invalid > "$output"\n')
    curl.chmod(0o755)
    result = invoke(workspace, "--toolchain-only")
    assert result.returncode != 0
    assert not sfpi.exists()
    assert not (wheel / "runtime").exists()
    assert not list((workspace / ".tools").glob(".sfpi-install.*"))
