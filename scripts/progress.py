"""Compatibility wrapper; the runner's status command also supports suites."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftbench_runner.status import show_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--human', action='store_true', help='Show a readable summary instead of JSON')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--progress-interval', type=float, default=15)
    args = parser.parse_args()
    if not 0 < args.progress_interval < float('inf'):
        parser.error('--progress-interval must be finite and positive')
    try:
        show_status(args.run_dir, not args.human, args.watch, args.progress_interval)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == '__main__':
    main()
