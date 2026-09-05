//===- execution.c - The C API, exercised the way a caller uses it --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C API is the one surface of this compiler with no MLIR on the other side
// of it, so nothing else here can test it: `tera-opt` and `tera-runner` link
// the libraries directly and never see the header. What does use it is a
// process in another language, and everything that has gone wrong at that
// boundary went wrong in C rather than in MLIR -- a result released by the
// wrong allocator, a pipeline that fails without a diagnostic, a descriptor
// whose extents nobody checked.
//
// So this test is a C program, linked against the shared library exactly as a
// caller links it, and it asks for what a caller asks for: an answer, the same
// module at a second batch, and a readable reason for each way a call can be
// wrong.
//
// RUN: tera-capi-test %mlir_c_runner_utils 2>&1 | FileCheck %s
// RUN: %if cuda %{ tera-capi-test %mlir_c_runner_utils %mlir_cuda_runtime 2>&1 | FileCheck %s --check-prefix=CUDA %}
//
//===----------------------------------------------------------------------===//

#include "Tera/CAPI/Execution.h"

#include <stdio.h>

static const char *kModule =
    "module @capi {\n"
    "  func.func @both(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>)"
    " -> (tensor<2x3xf32>, tensor<2x3xf32>) {\n"
    "    %s = tera.add %a, %b : tensor<2x3xf32>\n"
    "    %d = tera.sub %a, %b : tensor<2x3xf32>\n"
    "    return %s, %d : tensor<2x3xf32>, tensor<2x3xf32>\n"
    "  }\n"
    "  func.func @rowsum(%a: tensor<2x3xf32>) -> tensor<2xf32> {\n"
    "    %r = tera.reduce sum, %a {dimensions = array<i64: 1>}"
    " : tensor<2x3xf32> -> tensor<2xf32>\n"
    "    return %r : tensor<2xf32>\n"
    "  }\n"
    "  func.func @add2(%a: tensor<?x3xf32>, %b: tensor<?x3xf32>)"
    " -> tensor<?x3xf32> {\n"
    "    %r = tera.add %a, %b : tensor<?x3xf32>\n"
    "    return %r : tensor<?x3xf32>\n"
    "  }\n"
    "}\n";

/// A weight that stays on the device and a batch that does not, which is the
/// shape of every step of training and of inference: one argument uploaded
/// once and read every call, one arriving fresh each time.
static const char *kResidentModule =
    "module @resident {\n"
    "  func.func @scale(%w: tensor<2x3xf32> {tera.device_resident},"
    " %x: tensor<2x3xf32>) -> tensor<2x3xf32> {\n"
    "    %r = tera.mul %w, %x : tensor<2x3xf32>\n"
    "    return %r : tensor<2x3xf32>\n"
    "  }\n"
    "}\n";

/// A module no call can be made through, because a tensor of it cannot cross.
static const char *kNarrowModule =
    "module {\n"
    "  func.func @keep(%a: tensor<2xi8>) -> tensor<2xi8> {\n"
    "    return %a : tensor<2xi8>\n"
    "  }\n"
    "}\n";

/// Every number here is a multiple of 1/8, so each sum is exact in f32 and what
/// the checks below expect is the answer rather than a rounding of it.
static float kA[6] = {0.5f, -0.25f, 0.125f, 0.75f, 0.375f, -0.5f};
static float kB[6] = {0.25f, 0.5f, -0.125f, -0.25f, 0.125f, 0.25f};
static float kWideA[9] = {0.5f,  -0.25f, 0.125f, 0.75f, 0.375f,
                          -0.5f, 0.25f,  0.25f,  0.25f};
static float kWideB[9] = {0.25f, 0.5f,   -0.125f, -0.25f, 0.125f,
                          0.25f, 0.125f, -0.125f, 0.5f};

static void print(const char *label, const float *data, int64_t count) {
  printf("%s:", label);
  for (int64_t index = 0; index < count; index++)
    printf(" %g", data[index]);
  printf("\n");
}

static void expectFailure(const char *label, int status) {
  if (status == 0) {
    printf("%s: SUCCEEDED, which it should not have\n", label);
    return;
  }
  printf("%s: %s\n", label, teraLastError());
}

/// The device half of the API, which only a build with a device can run. The
/// caller allocates once, uploads once, and then calls with a pointer the host
/// cannot dereference -- so if the weight did not really stay there, the second
/// answer is wrong rather than merely slow.
static int deviceResident(const char *const *libraries, size_t numLibraries) {
  TeraModule *module =
      teraCompileFor(kResidentModule, "cuda", "", 3, libraries, numLibraries);
  if (!module) {
    printf("resident compile failed: %s\n", teraLastError());
    return 1;
  }

  void *weights = teraDeviceAlloc(module, sizeof(kA));
  if (!weights) {
    printf("device alloc failed: %s\n", teraLastError());
    return 1;
  }
  if (teraDeviceUpload(module, weights, kA, sizeof(kA)) != 0) {
    printf("upload failed: %s\n", teraLastError());
    return 1;
  }

  float scaled[6] = {0};
  int64_t shapes[4] = {2, 3, 2, 3};
  int64_t resultShapes[2] = {2, 3};
  for (int call = 0; call < 2; call++) {
    void *inputs[2] = {weights, call == 0 ? (void *)kB : (void *)kA};
    void *results[1] = {scaled};
    if (teraInvoke(module, "scale", inputs, shapes, 2, results, resultShapes,
                   1) != 0) {
      printf("scale failed: %s\n", teraLastError());
      return 1;
    }
    print(call == 0 ? "resident first" : "resident second", scaled, 6);
  }
  // CUDA: resident first: 0.125 -0.125 -0.015625 -0.1875 0.046875 -0.125
  // CUDA: resident second: 0.25 0.0625 0.015625 0.5625 0.140625 0.25

  // The weights are still where they were put, which is the whole claim.
  float readBack[6] = {0};
  if (teraDeviceDownload(module, readBack, weights, sizeof(kA)) != 0) {
    printf("download failed: %s\n", teraLastError());
    return 1;
  }
  print("resident weights", readBack, 6);
  // CUDA: resident weights: 0.5 -0.25 0.125 0.75 0.375 -0.5

  teraDeviceFree(module, weights);
  teraRelease(module);
  printf("resident released\n");
  // CUDA: resident released
  return 0;
}

int main(int argc, char **argv) {
  const char *libraries[2];
  size_t numLibraries = 0;
  for (int index = 1; index < argc && numLibraries < 2; index++)
    libraries[numLibraries++] = argv[index];

  TeraModule *module =
      teraCompileFor(kModule, "cpu", "", 3, libraries, numLibraries);
  if (!module) {
    printf("compile failed: %s\n", teraLastError());
    return 1;
  }
  printf("compiled\n");
  // CHECK: compiled

  void *pair[2] = {kA, kB};
  void *wideInputs[2] = {kWideA, kWideB};
  int64_t pairShapes[4] = {2, 3, 2, 3};
  int64_t wideShapes[4] = {3, 3, 3, 3};

  // Two results come back as one struct of descriptors laid end to end, which
  // is the shape a caller gets wrong first.
  float sum[6] = {0};
  float difference[6] = {0};
  {
    void *results[2] = {sum, difference};
    int64_t resultShapes[4] = {2, 3, 2, 3};
    if (teraInvoke(module, "both", pair, pairShapes, 2, results, resultShapes,
                   2) != 0) {
      printf("both failed: %s\n", teraLastError());
      return 1;
    }
  }
  print("sum", sum, 6);
  print("difference", difference, 6);
  // CHECK: sum: 0.75 0.25 0 0.5 0.5 -0.25
  // CHECK: difference: 0.25 -0.75 0.25 1 0.25 -0.75

  // A result of a lower rank than the argument, so a descriptor whose length is
  // read from the wrong signature is a wrong answer rather than one that agrees
  // by accident.
  float rows[2] = {0};
  {
    void *inputs[1] = {kA};
    void *results[1] = {rows};
    int64_t inputShapes[2] = {2, 3};
    int64_t resultShapes[1] = {2};
    if (teraInvoke(module, "rowsum", inputs, inputShapes, 1, results,
                   resultShapes, 1) != 0) {
      printf("rowsum failed: %s\n", teraLastError());
      return 1;
    }
  }
  print("rowsum", rows, 2);
  // CHECK: rowsum: 0.375 0.625

  // One compiled function at two batches. The extent comes with the call, and
  // that is the whole of what a dynamic dimension buys.
  float wide[9] = {0};
  {
    void *results[1] = {wide};
    int64_t resultShapes[2] = {2, 3};
    if (teraInvoke(module, "add2", pair, pairShapes, 2, results, resultShapes,
                   1) != 0) {
      printf("add2 failed: %s\n", teraLastError());
      return 1;
    }
    print("add2 two rows", wide, 6);
  }
  // CHECK: add2 two rows: 0.75 0.25 0 0.5 0.5 -0.25
  {
    void *results[1] = {wide};
    int64_t resultShapes[2] = {3, 3};
    if (teraInvoke(module, "add2", wideInputs, wideShapes, 2, results,
                   resultShapes, 1) != 0) {
      printf("add2 failed: %s\n", teraLastError());
      return 1;
    }
    print("add2 three rows", wide, 9);
  }
  // CHECK: add2 three rows: 0.75 0.25 0 0.5 0.5 -0.25 0.375 0.125 0.75

  // A result is the callee's allocation and is released on the way out. Neither
  // a leak nor a release through the wrong allocator shows in one call, so the
  // loop is the test: the heap either survives it or the process does not.
  for (int iteration = 0; iteration < 512; iteration++) {
    void *results[1] = {wide};
    int64_t resultShapes[2] = {2, 3};
    if (teraInvoke(module, "add2", pair, pairShapes, 2, results, resultShapes,
                   1) != 0) {
      printf("add2 failed on iteration %d: %s\n", iteration, teraLastError());
      return 1;
    }
  }
  printf("repeated calls survived\n");
  // CHECK: repeated calls survived

  {
    void *results[1] = {wide};
    int64_t resultShapes[2] = {2, 3};
    expectFailure("unknown entry",
                  teraInvoke(module, "nope", pair, pairShapes, 2, results,
                             resultShapes, 1));
    // CHECK: unknown entry: {{.*}}no function named nope

    expectFailure("wrong arity",
                  teraInvoke(module, "both", pair, pairShapes, 2, results,
                             resultShapes, 1));
    // CHECK: wrong arity: both takes 2 tensors and returns 2, but the call passes 2 and 1
  }

  // The caller says what shape it expects a result to have, because it sized
  // the buffer from that. A disagreement is the two shape inferences parting
  // company, and copying anyway would be writing past the end of the buffer.
  {
    void *results[1] = {wide};
    int64_t resultShapes[2] = {9, 3};
    expectFailure("wrong result shape",
                  teraInvoke(module, "add2", pair, pairShapes, 2, results,
                             resultShapes, 1));
    // CHECK: wrong result shape: add2 result 0 came back with a shape the caller did not expect
  }

  // Refused where it can still be explained, rather than at a call that has
  // already been handed the memory.
  if (teraCompileFor(kNarrowModule, "cpu", "", 3, libraries,
                     numLibraries)) {
    printf("a narrow element type compiled, which it should not have\n");
    return 1;
  }
  printf("narrow element type: %s\n", teraLastError());
  // CHECK: narrow element type: {{.*}}keep cannot be called from here
  // CHECK-NEXT: keep argument 0 has element type i8; only a float of at most 64 bits, i32 and i64 cross the JIT boundary

  // A second module open beside the first, to say that a handle is a handle and
  // not a name for global state.
  TeraModule *second =
      teraCompileFor(kModule, "cpu", "", 3, libraries, numLibraries);
  if (!second) {
    printf("second compile failed: %s\n", teraLastError());
    return 1;
  }
  {
    void *results[1] = {wide};
    int64_t resultShapes[2] = {2, 3};
    if (teraInvoke(second, "add2", pair, pairShapes, 2, results, resultShapes,
                   1) != 0) {
      printf("second module failed: %s\n", teraLastError());
      return 1;
    }
    print("second module", wide, 6);
  }
  // CHECK: second module: 0.75 0.25 0 0.5 0.5 -0.25
  teraRelease(second);

  // The host target has no device memory to hand out, and says so rather than
  // answering with a pointer that faults on the first read.
  if (teraDeviceAlloc(module, sizeof(kA))) {
    printf("the host target handed out device memory\n");
    return 1;
  }
  printf("no device memory: %s\n", teraLastError());
  // CHECK: no device memory: {{.*}}target with no device memory

  teraRelease(module);
  printf("released\n");
  // CHECK: released

  if (numLibraries > 1)
    return deviceResident(libraries, numLibraries);
  return 0;
}
