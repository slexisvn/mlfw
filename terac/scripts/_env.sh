# Every variable below can be preset in the environment. The defaults are only
# used when it is not, so another checkout or another LLVM is a variable to set
# rather than a line to edit.

TERAC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERAC_SIBLING="$(cd "${TERAC}/../.." && pwd)"

: "${LLVM_SRC:=${TERAC_SIBLING}/llvm-project}"
: "${LLVM_BUILD:=${LLVM_SRC}/build-assert}"
: "${MLIR_DIR:=${LLVM_BUILD}/lib/cmake/mlir}"
: "${NINJA:=$(command -v ninja || true)}"
: "${BUILD_TYPE:=Release}"

if [ ! -d "${MLIR_DIR}" ]; then
  echo "error: no MLIR cmake package at ${MLIR_DIR}." >&2
  echo "       Set MLIR_DIR to yours, or LLVM_BUILD to the build holding it." >&2
  exit 1
fi
if [ -z "${NINJA}" ]; then
  echo "error: ninja not found. Put it on PATH or set NINJA." >&2
  exit 1
fi

# MLIR_DIR is <prefix>/lib/cmake/mlir either way, so the tools and the runtime
# helpers are found from it rather than from a build layout this may not have.
LLVM_PREFIX="$(cd "${MLIR_DIR}/../../.." && pwd)"
: "${LLVM_TOOLS:=${LLVM_PREFIX}/bin}"
: "${LLVM_LIBS:=${LLVM_PREFIX}/lib}"

LLVM_LIT=""
for candidate in "${LLVM_TOOLS}/llvm-lit.py" "${LLVM_TOOLS}/llvm-lit"; do
  if [ -f "${candidate}" ]; then
    LLVM_LIT="${candidate}"
    break
  fi
done

export TERAC LLVM_SRC LLVM_BUILD MLIR_DIR NINJA BUILD_TYPE
export LLVM_PREFIX LLVM_TOOLS LLVM_LIBS LLVM_LIT
