#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/_env.sh"

runs="${1:-20}"
runner="${TERAC}/build/bin/tera-runner"
opt="${TERAC}/build/bin/tera-opt"

libs=""
for candidate in "${LLVM_LIBS}/libmlir_c_runner_utils.so" \
                 "${LLVM_LIBS}/libmlir_c_runner_utils.dylib" \
                 "${LLVM_TOOLS}/mlir_c_runner_utils.dll"; do
  if [ -f "${candidate}" ]; then
    libs="${candidate}"
    break
  fi
done
if [ -z "${libs}" ]; then
  echo "error: no mlir_c_runner_utils beside ${LLVM_PREFIX}." >&2
  exit 1
fi

if [ ! -x "${runner}" ]; then
  echo "Not built yet. Run scripts/build.sh first." >&2
  exit 1
fi

for model in mlp attention rnn; do
  source="${TERAC}/test/Integration/bench/${model}.mlir"
  "${runner}" "${source}" --entry="${model}" --benchmark="${runs}" \
      --shared-libs="${libs}"
  "${opt}" "${source}" --tera-autodiff -o "${TERAC}/build/${model}.vjp.mlir"
  "${runner}" "${TERAC}/build/${model}.vjp.mlir" --entry="${model}_vjp" \
      --benchmark="${runs}" --shared-libs="${libs}"
done
