#!/usr/bin/env python3
"""Exercise the longest published prompt through the normal HTTP inference path."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftbench_runner.inference import prepare
from driftbench_runner.http_inference import run_http


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _, rows, sources = prepare(args.config, ["long_context"])
    row = max(rows, key=lambda r: len(r["input_ids"]))
    required = len(row["input_ids"]) + config["generation"]["max_tokens"]
    if required > config["engine"]["max_model_len"]:
        raise ValueError(f"Longest request needs {required} tokens")
    sources["long_context"].update(selected=1, selection="longest_tokenized_input")
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Testing {row['prompt_id']}: {len(row['input_ids'])} input tokens, {required} total budget", flush=True)
    with (outdir / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_http(config, [row], sources, outdir)


if __name__ == "__main__":
    main()
