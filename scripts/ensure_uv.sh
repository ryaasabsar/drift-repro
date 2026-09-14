#!/usr/bin/env bash
# Shared workspace-local uv bootstrap; safe to source from installers.
set -euo pipefail
DRIFTBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$DRIFTBENCH_ROOT/.tools/uv" ]]; then
  mkdir -p "$DRIFTBENCH_ROOT/.tools"
  curl -fsSL https://astral.sh/uv/0.12.11/install.sh -o "$DRIFTBENCH_ROOT/.tools/install-uv.sh"
  UV_UNMANAGED_INSTALL="$DRIFTBENCH_ROOT/.tools" sh "$DRIFTBENCH_ROOT/.tools/install-uv.sh"
fi
