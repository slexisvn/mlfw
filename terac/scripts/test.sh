#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/_env.sh"

cmake --build "${TERAC}/build" --target check-tera
