#!/usr/bin/env bash
# Week 1 generation on one accelerator: a fresh vLLM server per repeat, batch size 1.
#   bash normbench-rework/run.sh a100|mi210|p150b [--repeats 3] [--out DIR] [--limit N | --percent P] [--port 8001]
# Override interpreters with SERVER_PYTHON; pass extra vLLM flags with EXTRA_ARGS; LOGPROBS=0 turns log-probs off.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
device="${1:?usage: run.sh a100|mi210|p150b [--repeats N] [--out DIR] [--limit N] [--port P]}"; shift
repeats=3; out="$here/results/$device"; limit=(); port=8001
while (($#)); do
  case "$1" in
    --repeats) repeats="$2"; shift 2;;
    --out) out="$2"; shift 2;;
    --limit) limit=(--limit "$2"); shift 2;;
    --percent) limit=(--percent "$2"); shift 2;;
    --port) port="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done

model=meta-llama/Llama-3.2-1B-Instruct
revision=9213176726f574b556790deb65791e0c5aa438b6
# Settings copied from configs/*-llama32-1b-instruct-vllm-http.json, except
# --max-num-seqs 1 (serial) and prefix caching off on every device.
common=(--model "${MODEL_PATH:-$model}" --revision "$revision" --tokenizer-revision "$revision"
        --served-model-name "$model" --generation-config vllm --seed 42
        --max-model-len 32768 --max-num-seqs 1 --no-enable-prefix-caching
        --host 127.0.0.1 --port "$port")
client=(--logprobs "${LOGPROBS:-5}")  # LOGPROBS=0 disables log-probs (GPU check)
export PYTHONHASHSEED=42
# Same workspace caches as the DriftBench runner (weights already downloaded there are reused).
export HF_HOME="${HF_HOME:-$root/.cache/huggingface}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/.cache}" \
  VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$root/.cache/vllm}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$root/.cache/triton}" \
  TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$root/.cache/torchinductor}"
# Llama is gated: when the pinned snapshot is already cached, never contact the Hub.
if [[ -d "$HF_HOME/hub/models--${model//\//--}/snapshots/$revision" ]]; then export HF_HUB_OFFLINE=1; fi
case "$device" in
  a100)
    SERVER_PYTHON="${SERVER_PYTHON:-$root/.venv/bin/python}"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    engine=(--dtype bfloat16 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8
            --tensor-parallel-size 1 --no-enforce-eager)
    timeout=900;;
  mi210)
    SERVER_PYTHON="${SERVER_PYTHON:-$root/.venv-rocm-vllm/bin/python}"
    # Ray/vLLM want a HIP mask inside a single-GPU ROCr allocation.
    if [[ -z "${HIP_VISIBLE_DEVICES:-}" && "${ROCR_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then export HIP_VISIBLE_DEVICES=0; fi
    engine=(--dtype bfloat16 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8
            --tensor-parallel-size 1 --no-enforce-eager)
    timeout=900;;
  p150b)
    SERVER_PYTHON="${SERVER_PYTHON:-$root/.venv-tt-vllm/bin/python}"
    if [[ -f "$root/.venv-tt-vllm/activate-tt.sh" ]]; then source "$root/.venv-tt-vllm/activate-tt.sh"; fi
    export MESH_DEVICE=P150 ARCH_NAME=blackhole TORCHDYNAMO_DISABLE=1 VLLM_CONFIGURE_LOGGING=1
    engine=(--max-num-batched-tokens 32768 --block-size 64
            --additional-config '{"tt":{"sample_on_device_mode":"all"}}')
    # On-device sampling returns neither token IDs nor log-probs; the report compares text.
    client=(--no-token-ids --logprobs 0)
    timeout=3600;;
  *) echo "device must be a100, mi210 or p150b" >&2; exit 2;;
esac
read -r -a extra <<< "${EXTRA_ARGS:-}"

server_pid=
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid"; for _ in $(seq 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
trap stop_server EXIT

for r in $(seq 1 "$repeats"); do
  dir="$out/r$r"; mkdir -p "$dir"
  if [[ -f "$dir/DONE" ]]; then echo "skip $dir (DONE)"; continue; fi
  echo "== $device repeat $r: starting fresh server (log: $dir/server.log)"
  "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server "${common[@]}" "${engine[@]}" "${extra[@]}" \
    >> "$dir/server.log" 2>&1 &
  server_pid=$!
  for ((t = 0; ; t += 5)); do
    if curl -sf "http://127.0.0.1:$port/health" > /dev/null; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then echo "server exited; see $dir/server.log" >&2; exit 1; fi
    if ((t >= timeout)); then echo "server not ready after ${timeout}s" >&2; exit 1; fi
    sleep 5
  done
  printf '%q ' "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server "${common[@]}" "${engine[@]}" "${extra[@]}" \
    > "$dir/server-command.txt"
  "$SERVER_PYTHON" "$here/week1.py" generate --device "$device" --repeat "$r" --out "$dir" \
    --url "http://127.0.0.1:$port" "${limit[@]}" "${client[@]}"
  stop_server
  if ((${#limit[@]} == 0)); then touch "$dir/DONE"; fi  # a --limit/--percent run never marks a repeat complete
done
echo "done: $out"
