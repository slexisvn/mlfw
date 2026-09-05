//===- Execution.h - Calling the tera JIT from another language ---*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

typedef struct TeraModule TeraModule;

enum {
  TERA_TARGET_CPU = 0,
  TERA_TARGET_CUDA = 1,
};

TERA_CAPI TeraModule *teraCompile(const char *mlir, int target,
                                  unsigned optLevel,
                                  const char *const *sharedLibs,
                                  size_t numSharedLibs);

TERA_CAPI TeraModule *teraCompileFor(const char *mlir, const char *target,
                                     const char *targetOptions,
                                     unsigned optLevel,
                                     const char *const *sharedLibs,
                                     size_t numSharedLibs);

TERA_CAPI const char *teraTargets(void);

TERA_CAPI const char *teraTargetRuntimeLibraries(const char *target);

TERA_CAPI void teraRelease(TeraModule *module);

TERA_CAPI const char *teraLastError(void);

TERA_CAPI int teraInvoke(TeraModule *module, const char *entry,
                         void *const *inputs, const int64_t *inputShapes,
                         int64_t numInputs, void *const *results,
                         const int64_t *resultShapes, int64_t numResults);

#ifdef __cplusplus
}
#endif

#endif
