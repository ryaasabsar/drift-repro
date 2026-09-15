"""Framework-specific wire protocols; no synthesized generated token IDs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import digest


class HTTPBackend:
    def __init__(self, config):
        self.config = config
        self.backend = config["backend"]
        self.server = config["server"]
        self.base_url = self.server["base_url"].rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) server root without credentials or query parameters")
        if parsed.path.rstrip("/"):
            raise ValueError("base_url must be the server root, without /v1")
        self.headers = {"Content-Type": "application/json"}
        if self.server.get("api_key_env"):
            key = os.environ.get(self.server["api_key_env"])
            if not key:
                raise ValueError(f"Set environment variable {self.server['api_key_env']} for server authentication")
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = float(self.server.get("timeout_seconds", 900))

    def request(self, path, payload=None):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, headers=self.headers,
                                         method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            # Do not include headers or credentials in error logs.
            raise RuntimeError(f"{self.backend} {path} returned HTTP {exc.code}: {exc.read(2000).decode(errors='replace')}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.backend} request to {path} failed: {exc.reason}") from None

    def probe(self):
        models = self.request("/v1/models")
        expected = self.server.get("model_name", self.config["model"])
        ids = [r["id"] for r in models.get("data", [])]
        if expected not in ids:
            raise ValueError(f"Server does not expose configured model {expected!r}; found {ids}")
        info = {"models": ids}
        if self.backend == "sglang":
            raw = self.request("/get_model_info")
            info["model_info"] = {k: raw[k] for k in ("model_path", "tokenizer_path", "is_generation", "weight_version", "model_type") if k in raw}
        return info

    def payload(self, row, max_tokens=None):
        generation = dict(self.config["generation"])
        budget = generation.pop("max_tokens") if max_tokens is None else max_tokens
        generation.pop("max_tokens", None)
        supported = {"temperature", "top_p", "top_k", "repetition_penalty",
                     "presence_penalty", "frequency_penalty", "min_p"}
        if set(generation) - supported:
            raise ValueError(f"Unmapped generation controls: {set(generation) - supported}")
        if self.backend == "sglang":
            params = {**generation, "max_new_tokens": budget, "sampling_seed": self.config["seed"]}
            return "/generate", {"input_ids": row["input_ids"], "sampling_params": params,
                                  "return_logprob": True, "logprob_start_len": -1, "stream": False}
        payload = {"model": self.server.get("model_name", self.config["model"]),
                   "prompt": row["input_ids"], "max_tokens": budget, "seed": self.config["seed"],
                   "n": 1, "stream": False, "add_special_tokens": False, **generation}
        if self.backend == "vllm" and self.server.get("return_token_ids", True):
            payload["return_token_ids"] = True
        return "/v1/completions", payload

    def generate_one(self, row, max_tokens=None):
        path, payload = self.payload(row, max_tokens)
        start = time.perf_counter()
        raw = self.request(path, payload)
        elapsed = time.perf_counter() - start
        if self.backend == "sglang":
            if not isinstance(raw, dict):
                raise ValueError("Expected one SGLang response per request")
            meta = raw["meta_info"]
            ids = raw.get("output_ids")
            if ids is None and meta.get("output_token_logprobs") is not None:
                ids = [item[1] for item in meta["output_token_logprobs"]]
            reason = meta.get("finish_reason", {})
            finish = reason.get("type") if isinstance(reason, dict) else reason
            finish = {"stop": "stop", "length": "length", "abort": "abort"}.get(finish, finish)
            result = {"output_text": raw["text"], "output_token_ids": ids,
                      "output_tokens": meta["completion_tokens"], "server_input_tokens": meta["prompt_tokens"],
                      "finish_reason": finish, "stop_reason": reason.get("matched") if isinstance(reason, dict) else None}
        else:
            choices = raw.get("choices", [])
            if len(choices) != 1 or choices[0].get("index", 0) != 0:
                raise ValueError("Expected exactly one completion for each input")
            choice = choices[0]
            result = {"output_text": choice["text"], "output_token_ids": choice.get("token_ids"),
                      "output_tokens": raw["usage"]["completion_tokens"],
                      "server_input_tokens": raw["usage"]["prompt_tokens"],
                      "finish_reason": choice["finish_reason"], "stop_reason": choice.get("stop_reason")}
            echoed = choice.get("prompt_token_ids", raw.get("prompt_token_ids"))
            if echoed is not None and echoed != row["input_ids"]:
                raise ValueError("Server changed the supplied input token IDs")
        if result["server_input_tokens"] != len(row["input_ids"]):
            raise ValueError("Server token count differs from submitted input; refusing possible truncation")
        ids = result["output_token_ids"]
        if ids is not None and (not all(type(t) is int and t >= 0 for t in ids) or len(ids) != result["output_tokens"]):
            raise ValueError("Generated token IDs and server completion count disagree")
        if result["finish_reason"] not in ("length", "stop"):
            raise ValueError(f"Generation did not complete normally: {result['finish_reason']}")
        budget = self.config["generation"]["max_tokens"] if max_tokens is None else max_tokens
        if not 0 <= result["output_tokens"] <= budget:
            raise ValueError("Server exceeded the requested generation budget")
        result.update(token_ids_source="server" if ids is not None else "unavailable",
                      latency_seconds=elapsed, server_response=raw, wire_request_sha256=digest(payload),
                      effective_sampling={k: v for k, v in payload.items() if k not in ("prompt", "input_ids")})
        return result

    def generate(self, batch):
        if len(batch) == 1:
            return [self.generate_one(batch[0])]
        # A barrier separates client batches; this is not a claim about GPU batching.
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            return list(pool.map(self.generate_one, batch))
