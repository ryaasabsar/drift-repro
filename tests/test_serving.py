"""Protocol fixtures test control preservation, not accelerator compatibility."""
import copy
import json

import pytest

from driftbench_runner.common import digest, read_json, write_json
from driftbench_runner.hardware import validate_target
from driftbench_runner.http_backend import HTTPBackend
from driftbench_runner.http_inference import run_http
from driftbench_runner.serving import launch_command, serving_identity


@pytest.fixture
def config(tmp_path):
    return {"setup_id": "fixture", "backend": "vllm", "transport": "http", "model": "fixture/model",
            "revision": "a" * 40, "seed": 42, "batch_size": 2,
            "hardware": {"vendor": "nvidia", "device": "fixture GPU"},
            "engine": {"dtype": "bfloat16", "max_model_len": 1024, "enable_prefix_caching": False},
            "generation": {"temperature": 0, "top_p": 1, "top_k": -1, "max_tokens": 8, "repetition_penalty": 1},
            "server": {"base_url": "http://127.0.0.1:8001", "metadata_path": str(tmp_path / "server.json")}}


def completion(ids=(3, 4), prompt_count=2):
    return {"choices": [{"index": 0, "text": "fixture answer", "token_ids": ids, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 2, "prompt_tokens": prompt_count}}


def test_vllm_transmits_exact_ids_and_explicit_controls(config, monkeypatch):
    backend = HTTPBackend(config)
    path, payload = backend.payload({"input_ids": [1, 2]})
    assert path == "/v1/completions"
    assert payload["prompt"] == [1, 2] and payload["add_special_tokens"] is False
    assert payload["seed"] == 42 and payload["return_token_ids"] is True
    monkeypatch.setattr(backend, "request", lambda *args: completion())
    result = backend.generate_one({"input_ids": [1, 2]})
    assert result["output_token_ids"] == (3, 4)
    assert result["token_ids_source"] == "server"


def test_sglang_extracts_actual_generated_ids(config, monkeypatch):
    config["backend"] = "sglang"
    backend = HTTPBackend(config)
    path, payload = backend.payload({"input_ids": [1, 2]})
    assert path == "/generate" and payload["sampling_params"]["sampling_seed"] == 42
    assert "prompt" not in payload and payload["return_logprob"]
    response = {"text": "fixture answer", "meta_info": {"prompt_tokens": 2, "completion_tokens": 2,
                "finish_reason": {"type": "stop", "matched": 4}, "output_token_logprobs": [[-1., 3, None], [-2., 4, None]]}}
    monkeypatch.setattr(backend, "request", lambda *args: response)
    assert backend.generate_one({"input_ids": [1, 2]})["output_token_ids"] == [3, 4]


def test_tensorrt_missing_token_ids_are_explicit(config, monkeypatch):
    config["backend"] = "tensorrt-llm"
    backend = HTTPBackend(config)
    _, payload = backend.payload({"input_ids": [1, 2]})
    assert payload["top_k"] == 0 and "return_token_ids" not in payload
    response = completion()
    response["choices"][0].pop("token_ids")
    monkeypatch.setattr(backend, "request", lambda *args: response)
    result = backend.generate_one({"input_ids": [1, 2]})
    assert result["output_token_ids"] is None and result["token_ids_source"] == "unavailable"


@pytest.mark.parametrize("response,match", [(completion(prompt_count=1), "token count"),
    (completion(ids=[3]), "completion count"),
    ({"choices": [], "usage": {}}, "exactly one")])
def test_bad_server_responses_fail(config, monkeypatch, response, match):
    backend = HTTPBackend(config)
    monkeypatch.setattr(backend, "request", lambda *args: response)
    with pytest.raises(ValueError, match=match):
        backend.generate_one({"input_ids": [1, 2]})


@pytest.mark.parametrize("vendor", ["amd", "tenstorrent"])
def test_tensorrt_rejects_non_nvidia(config, vendor):
    config.update(backend="tensorrt-llm")
    config["hardware"]["vendor"] = vendor
    with pytest.raises(ValueError, match="Unsupported"):
        validate_target(config)


def test_launch_rejects_unmapped_controls(config):
    config["engine"]["made_up_option"] = 5
    with pytest.raises(ValueError, match="Unmapped"):
        launch_command(config)


def test_launch_pins_weights_and_disables_prefix_cache(config):
    command = launch_command(config)
    assert command[command.index("--revision") + 1] == "a" * 40
    assert "--no-enable-prefix-caching" in command
    config["launch"] = {"extra_args": ["--revision", "main"]}
    with pytest.raises(ValueError, match="override"):
        launch_command(config)


def test_remote_host_is_for_client_only(config):
    config["server"]["base_url"] = "http://192.0.2.1:8001"
    HTTPBackend(config)
    with pytest.raises(ValueError, match="local HTTP"):
        launch_command(config)


def test_tensorrt_controls_include_pinned_revision_and_context_failure(config):
    config['backend'] = 'tensorrt-llm'
    config['engine'] = {'dtype': 'bfloat16', 'max_model_len': 24576, 'tensor_parallel_size': 1,
                        'enable_chunked_prefill': True, 'fail_fast_on_attention_window_too_large': True}
    command = launch_command(config)
    assert '--revision' in command and '--extra_llm_api_options' in command
    assert '--enable_chunked_prefill' in command
    assert '--fail_fast_on_attention_window_too_large' in command
    config['launch'] = {'extra_args': ['--config', '/tmp/untracked.yaml']}
    with pytest.raises(ValueError, match='override'):
        launch_command(config)


def test_http_manifest_resume_and_server_binding(config, tmp_path, monkeypatch):
    identity = serving_identity(config)
    metadata = {"identity": identity, "serving_fingerprint": digest(identity), "status": "ready",
                "instance_id": "fixture-instance", "environment": {"accelerators": [{"name": "server GPU"}]}}
    write_json(config["server"]["metadata_path"], metadata)
    monkeypatch.setattr(HTTPBackend, "probe", lambda _: {"models": [config["model"]]})
    monkeypatch.setattr(HTTPBackend, "request", lambda *args: completion())
    rows = [{"workload": "math", "prompt_id": str(i), "request_sha256": str(i), "source_sha256": str(i),
             "input_ids": [1, 2], "prompt": "fixture", "rendered_prompt": "fixture"} for i in range(3)]
    outdir = tmp_path / "run"
    outdir.mkdir()
    manifest = run_http(config, rows, {"math": {"selected": 3}}, outdir)
    assert manifest["status"] == "complete"
    assert manifest["environment"]["gpu_name"] == "server GPU"
    assert manifest["scheduling"] == "client_concurrent_barrier"
    assert run_http(config, rows, {"math": {"selected": 3}}, outdir, resume=True)["completed_records"] == 3
    metadata["environment"]["accelerators"][0]["name"] = "different GPU"
    write_json(config["server"]["metadata_path"], metadata)
    with pytest.raises(ValueError, match="serving environment changed"):
        run_http(config, rows, {"math": {"selected": 3}}, outdir, resume=True)
    config["revision"] = "b" * 40
    with pytest.raises(ValueError, match="provenance does not match"):
        run_http(config, rows, {"math": {"selected": 3}}, outdir, resume=True)


def test_actual_server_context_overrides_requested_capacity(config, tmp_path):
    identity = serving_identity(config)
    write_json(config['server']['metadata_path'], {'identity': identity, 'serving_fingerprint': digest(identity),
               'status': 'ready', 'effective_context_limit': 4})
    with pytest.raises(ValueError, match='effective context limit'):
        run_http(config, [{'input_ids': [1, 2]}], {}, tmp_path)
