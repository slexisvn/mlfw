//===- Execution.h - Calling the tera JIT from another language ---*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `tera-runner` is a driver: it compiles a module, calls it once and exits, and
// the tensors reach it as JSON. That is the right shape for a test and the
// wrong one for a caller that means to run the program, because the JIT is paid
// for on every call and every number is parsed twice.
//
// This is the same machinery with the driver taken off. `teraCompile` holds the
// engine open behind a handle, and `teraInvoke` calls into it with the caller's
// own memory: the descriptors are built over the pointers it hands in, so
// nothing is copied on the way in and only the results the callee allocated are
// copied on the way out.
//
// The contract is deliberately thin -- four functions, no types of its own but
// the opaque handle -- because everything it could have carried is already
// known to the caller that wrote the module. Ranks and element types come from
// the signature this remembers before lowering erases it; extents come with the
// call, which is what makes one compiled module answer for every batch.
//
//===----------------------------------------------------------------------===//

#ifndef TERA_CAPI_EXECUTION_H
#define TERA_CAPI_EXECUTION_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TERA_CAPI_BUILDING)
#define TERA_CAPI __declspec(dllexport)
#else
#define TERA_CAPI __declspec(dllimport)
#endif
#else
#define TERA_CAPI __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// A parsed, lowered and JIT-compiled module, and the signatures its entries
/// had before the lowering rewrote them into pointers and extents.
typedef struct TeraModule TeraModule;

enum {
  TERA_TARGET_CPU = 0,
  TERA_TARGET_CUDA = 1,
};

/// Compiles \p mlir for \p target; release the handle with teraRelease.
/// Runtime libraries must include mlir_c_runner_utils and, for CUDA,
/// mlir_cuda_runtime. Returns null on failure; see teraLastError.
TERA_CAPI TeraModule *teraCompile(const char *mlir, int target,
                                  unsigned optLevel,
                                  const char *const *sharedLibs,
                                  size_t numSharedLibs);

/// Releases the compiled module and its JIT engine.
TERA_CAPI void teraRelease(TeraModule *module);

/// Returns the last error on this thread, valid until the next call.
TERA_CAPI const char *teraLastError(void);

/// Invokes \p entry using caller-owned, contiguous row-major tensor buffers.
/// Shape arrays concatenate tensor extents in signature order. Results are
/// copied into the supplied buffers; mismatched result shapes are errors.
/// Returns 0 on success or -1 on failure; see teraLastError.
TERA_CAPI int teraInvoke(TeraModule *module, const char *entry,
                         void *const *inputs, const int64_t *inputShapes,
                         int64_t numInputs, void *const *results,
                         const int64_t *resultShapes, int64_t numResults);

#ifdef __cplusplus
}
#endif

#endif // TERA_CAPI_EXECUTION_H
