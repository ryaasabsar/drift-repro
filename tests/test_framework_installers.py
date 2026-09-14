import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("installer", ROOT / "scripts/install_framework.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.mark.parametrize("framework", ["sglang", "tensorrt"])
def test_dry_run_from_another_directory_creates_nothing(tmp_path, framework):
    target = installer.FRAMEWORKS[framework].venv + "-dry-run-test"
    assert not (ROOT / target).exists()
    result = subprocess.run(["bash", str(ROOT / f"scripts/install_{framework}.sh"),
                             "--venv", target, "--dry-run"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"requirements.{framework}.lock.txt" in result.stdout
    assert "--no-deps" in result.stdout
    assert not (ROOT / target).exists()


@pytest.mark.parametrize("target", [".venv", ".venv-rocm-sglang", "../.venv-sglang", "/usr"])
def test_installer_refuses_unrelated_environments(tmp_path, monkeypatch, target):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        installer.target_path("sglang", target)


def test_installer_refuses_symlink_and_non_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    target = tmp_path / ".venv-sglang"
    target.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        installer.target_path("sglang", None)
    target.unlink()
    target.mkdir()
    with pytest.raises(ValueError):
        installer.target_path("sglang", None)


def test_existing_environment_is_reused_and_failed_sync_stops_install(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    target = tmp_path / ".venv-sglang"
    target.mkdir()
    (target / "pyvenv.cfg").write_text(f"version = 3.12.14\nhome = {tmp_path}/.tools/python/cpython-3.12/bin\n")
    monkeypatch.setattr(sys, "argv", ["installer", "sglang"])
    commands = []

    def fail_sync(command, *args, **kwargs):
        commands.append([str(x) for x in command])
        if "sync" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(installer, "run", fail_sync)
    assert installer.main() == 1
    assert not any("venv" in command for command in commands)
    assert "sync" in commands[-1]
    assert (target / "pyvenv.cfg").exists()


def test_check_missing_environment_does_not_install(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["installer", "tensorrt", "--check"])
    assert installer.main() == 1
    assert list(tmp_path.iterdir()) == []


def test_system_python_is_replaced_in_same_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    target = tmp_path / ".venv-sglang"
    target.mkdir()
    (target / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.12.3\n")
    monkeypatch.setattr(sys, "argv", ["installer", "sglang", "--dry-run"])
    commands = []
    monkeypatch.setattr(installer, "run", lambda command, *args, **kwargs: commands.append([str(x) for x in command]))
    assert installer.main() == 0
    creation = next(c for c in commands if "venv" in c)
    assert "--managed-python" in creation and "--clear" in creation
    assert creation[-1] == str(target)
    assert (target / "pyvenv.cfg").read_text().startswith("home = /usr/bin")


@pytest.mark.parametrize("check_gpu", [False, True])
def test_verification_uses_target_interpreter_and_gpu_check_is_explicit(monkeypatch, check_gpu):
    commands = []
    monkeypatch.setattr(installer, "run", lambda command, *args, **kwargs: commands.append(command))
    installer.verify("sglang", ROOT / ".venv-sglang", {}, check_gpu=check_gpu)
    probe = commands[-1]
    assert probe[:3] == [ROOT / ".venv-sglang/bin/python", "-m", "driftbench_runner.framework_check"]
    assert ("--check-gpu" in probe) == check_gpu
