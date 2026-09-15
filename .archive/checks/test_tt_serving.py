"""Verify TT weight selection without starting vLLM or opening a device."""
import json

import huggingface_hub
import pytest

from driftbench_runner import serving, software
from driftbench_runner.common import ROOT, read_json


@pytest.mark.parametrize("profile", ["qwen25-7b", "qwen35-9b-base", "llama31-8b-instruct"])
def test_tt_loader_uses_pinned_snapshot_and_revision_cache(tmp_path, monkeypatch, profile):
    config = read_json(ROOT / "configs" / f"blackhole-p150b-{profile}-vllm-http.json")
    config["server"]["metadata_path"] = str(tmp_path / "server.json")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    environment = {"packages": {"vllm": "0.25.1+empty"}, "runtime_environment": {}}
    monkeypatch.setattr(serving, "ROOT", tmp_path)
    monkeypatch.setattr(serving, "local_environment", lambda: None)
    monkeypatch.setattr(serving, "configure_serving_environment", lambda _: None)
    monkeypatch.setattr(serving, "discover", lambda: environment)
    monkeypatch.setattr(serving, "require_hardware", lambda *_: None)
    monkeypatch.setattr(software, "check_runtime", lambda *_: {"status": "match"})
    monkeypatch.setenv("HF_MODEL", "wrong/model")
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(tmp_path / "wrong-weights"))
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "wrong-cache"))

    def download(model, revision):
        assert (model, revision) == (config["model"], config["revision"])
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    class UnusedPort:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def connect_ex(self, _):
            return 1

    monkeypatch.setattr(serving.socket, "socket", lambda *_: UnusedPort())
    captured = {}

    def capture_launch(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        raise RuntimeError("stop before accelerator execution")

    monkeypatch.setattr(serving.subprocess, "Popen", capture_launch)
    with pytest.raises(RuntimeError, match="stop before accelerator execution"):
        serving.serve(config_path)

    alias = serving.model_snapshot_alias(config)
    assert alias.resolve() == snapshot
    assert captured["env"]["HF_MODEL"] == str(alias)
    assert captured["env"]["MODEL_WEIGHTS_DIR"] == str(alias)
    assert str(alias) in captured["command"]
    cache = tmp_path / ".cache/tt-models" / config["revision"] / config["model"].split("/")[-1]
    assert captured["env"]["TT_CACHE_PATH"] == str(cache)
    recorded = read_json(config["server"]["metadata_path"])["environment"]["runtime_environment"]
    assert recorded["HF_MODEL"] == str(alias)
    assert recorded["MODEL_WEIGHTS_DIR"] == str(alias)
    assert recorded["TT_CACHE_PATH"] == str(cache)
