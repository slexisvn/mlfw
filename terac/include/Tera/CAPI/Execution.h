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

/* Builds `mlir` for the named target and answers a handle to call into, or
 * null with teraLastError saying why.
 *
 * `target` is a name from teraTargets and `targetOptions` is that target's own
 * `name=value` pairs, empty for its defaults. Naming the target rather than
 * numbering it is what lets a caller reach a target this header has never
 * heard of, and a target's options at all: an enum would have to be extended
 * here and recompiled on both sides for either.
 */
TERA_CAPI TeraModule *teraCompileFor(const char *mlir, const char *target,
                                     const char *targetOptions,
                                     unsigned optLevel,
                                     const char *const *sharedLibs,
                                     size_t numSharedLibs);

/* The targets teraCompileFor accepts, comma separated. */
TERA_CAPI const char *teraTargets(void);

/* The runtime libraries a target's modules must be given in `sharedLibs`,
 * comma separated, as stems without a prefix or an extension. */
TERA_CAPI const char *teraTargetRuntimeLibraries(const char *target);

TERA_CAPI void teraRelease(TeraModule *module);

TERA_CAPI const char *teraLastError(void);

TERA_CAPI int teraInvoke(TeraModule *module, const char *entry,
                         void *const *inputs, const int64_t *inputShapes,
                         int64_t numInputs, void *const *results,
                         const int64_t *resultShapes, int64_t numResults);

/* Memory on the device the module was built for, which outlives one call.
 *
 * A pointer from teraDeviceAlloc is passed to teraInvoke in place of a host
 * one, for an argument the module marks `tera.device_resident`. That is how a
 * weight is uploaded once and read every call instead of crossing each time.
 * The compiled entry point stages every other argument as before, so the two
 * kinds can be mixed in one call.
 *
 * teraDeviceAlloc answers null, and the copies answer non-zero, when the
 * module was built for a target with no device memory; teraLastError says so.
 */
TERA_CAPI void *teraDeviceAlloc(TeraModule *module, size_t bytes);

TERA_CAPI void teraDeviceFree(TeraModule *module, void *pointer);

TERA_CAPI int teraDeviceUpload(TeraModule *module, void *device,
                               const void *host, size_t bytes);

TERA_CAPI int teraDeviceDownload(TeraModule *module, void *host,
                                 const void *device, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif
