#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/_env.sh"

lit=()
if [ -n "${LLVM_LIT}" ]; then
  lit=(-DLLVM_EXTERNAL_LIT="${LLVM_LIT}")
fi

cmake -G Ninja \
  -S "${TERAC}" \
  -B "${TERAC}/build" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DMLIR_INCLUDE_TESTS=ON \
  "${lit[@]}" \
  -DCMAKE_MAKE_PROGRAM="${NINJA}"
