#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/_env.sh"

if [ ! -f "${TERAC}/build/build.ninja" ]; then
  echo "Not configured yet. Run scripts/configure.sh first." >&2
  exit 1
fi
cmake --build "${TERAC}/build"
