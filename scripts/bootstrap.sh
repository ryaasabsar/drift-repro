#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools vendor
if [[ ! -x .tools/uv ]]; then
  curl -fsSL https://astral.sh/uv/install.sh -o .tools/install-uv.sh
  UV_UNMANAGED_INSTALL="$PWD/.tools" sh .tools/install-uv.sh
fi
source scripts/env.sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
if [[ ! -d vendor/driftbench-ae ]]; then
  git clone https://github.com/GianluigiVitale/driftbench-ae.git vendor/driftbench-ae
fi
git -C vendor/driftbench-ae checkout c915e781a17c2d4c9bd768e60a1bc9734c5f2895
if ! command -v bwrap >/dev/null && [[ ! -x .tools/bubblewrap-root/usr/bin/bwrap ]]; then
  if command -v apt-get >/dev/null; then
    (cd .tools && apt-get download bubblewrap && dpkg-deb -x bubblewrap_*.deb bubblewrap-root)
  else
    echo 'Install bubblewrap with your Linux package manager to enable isolated HumanEval evaluation.'
  fi
fi
echo 'Ready. Run: source scripts/env.sh'
