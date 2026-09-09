#!/usr/bin/env bash
# Run from the project root after activating the benchmark client's environment.
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: bash scripts/run_served.sh SERVER_PYTHON CONFIG OUTPUT [run arguments, e.g. --limit 2]" >&2
  exit 2
fi
DRIFT_SERVER_PYTHON="$1"
DRIFT_CONFIG="$2"
DRIFT_OUTPUT="$3"
shift 3
mkdir -p "$DRIFT_OUTPUT"
DRIFT_STARTED="$(python -c 'import time; print(time.time())')"
"$DRIFT_SERVER_PYTHON" -m driftbench_runner serve --config "$DRIFT_CONFIG" > "$DRIFT_OUTPUT/launcher.log" 2>&1 &
DRIFT_SERVER_PID=$!
cleanup() {
  if kill -0 "$DRIFT_SERVER_PID" 2>/dev/null; then
    kill -TERM "$DRIFT_SERVER_PID"
    wait "$DRIFT_SERVER_PID" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python - "$DRIFT_CONFIG" "$DRIFT_STARTED" "$DRIFT_SERVER_PID" <<'PY'
import datetime, json, os, sys, time
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text())
path=Path(config['server']['metadata_path'])
started=float(sys.argv[2]); pid=int(sys.argv[3])
deadline=time.time()+config.get('launch',{}).get('startup_timeout_seconds',900)+config['server'].get('timeout_seconds',900)
while time.time()<deadline:
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        raise SystemExit('Server exited; inspect launcher.log and the server log')
    if path.exists():
        m=json.loads(path.read_text())
        if datetime.datetime.fromisoformat(m['created_at']).timestamp()>=started:
            if m['status']=='ready': break
            if m['status']=='failed': raise SystemExit(m.get('error','Server failed'))
    time.sleep(2)
else:
    raise SystemExit('Timed out waiting for server')
PY
python -m driftbench_runner run --config "$DRIFT_CONFIG" --output "$DRIFT_OUTPUT" "$@"
cleanup
trap - EXIT INT TERM
# Evaluate on this common host only after the inference server releases its GPU.
if python - "$DRIFT_OUTPUT/manifest.json" <<'PY'
import json,sys
sys.exit(0 if 'safety' in json.load(open(sys.argv[1]))['sources'] else 1)
PY
then
  python -m driftbench_runner judge-safety "$DRIFT_OUTPUT" --device cpu --output "$DRIFT_OUTPUT/safety-labels.jsonl"
  python -m driftbench_runner evaluate "$DRIFT_OUTPUT" --code --safety-labels "$DRIFT_OUTPUT/safety-labels.jsonl"
else
  python -m driftbench_runner evaluate "$DRIFT_OUTPUT" --code
fi
