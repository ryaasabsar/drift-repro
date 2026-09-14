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
mkdir -p results
exec 9>results/.gpu-experiment.lock
flock -n 9 || { echo "Another GPU experiment is running" >&2; exit 1; }
mkdir -p "$DRIFT_OUTPUT"
mkdir -p "$DRIFT_OUTPUT/logs"
DRIFT_STARTED="$(python -c 'import time; print(time.time())')"
DRIFTBENCH_EVENTS_FILE="$DRIFT_OUTPUT/logs/launcher.events.jsonl" \
  "$DRIFT_SERVER_PYTHON" -u -m driftbench_runner serve --config "$DRIFT_CONFIG" > "$DRIFT_OUTPUT/launcher.log" 2>&1 &
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
python - "$DRIFT_CONFIG" "$DRIFT_STARTED" "$DRIFT_SERVER_PID" "$DRIFT_OUTPUT" <<'PY'
import datetime, json, os, sys, time
from pathlib import Path
from driftbench_runner.logging import EventFollower, activity, configure
config=json.loads(Path(sys.argv[1]).read_text())
path=Path(config['server']['metadata_path'])
started=float(sys.argv[2]); pid=int(sys.argv[3])
directory=Path(sys.argv[4])
configure(os.environ.get('DRIFTBENCH_LOG_LEVEL', 'info'), directory/'logs/startup.log',
          directory/'logs/startup.events.jsonl', float(os.environ.get('DRIFTBENCH_PROGRESS_INTERVAL', '15')))
follower=EventFollower(directory/'logs/launcher.events.jsonl', since=started)
deadline=time.time()+config.get('launch',{}).get('startup_timeout_seconds',900)+config['server'].get('timeout_seconds',900)
with activity('Waiting for serving launcher', stage='startup', setting=config['setup_id'], log=str(directory/'launcher.log')):
    while time.time()<deadline:
        follower.drain()
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            raise SystemExit('Server exited; inspect launcher.log and the server log')
        if path.exists():
            m=json.loads(path.read_text())
            if datetime.datetime.fromisoformat(m['created_at']).timestamp()>=started and m.get('launcher_pid')==pid:
                if m['status']=='ready': break
                if m['status']=='failed': raise SystemExit(m.get('error','Server failed'))
        time.sleep(2)
    else:
        raise SystemExit('Timed out waiting for server')
    follower.drain()
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
