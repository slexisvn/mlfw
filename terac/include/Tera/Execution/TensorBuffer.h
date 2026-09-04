//===- TensorBuffer.h - A tensor across the JIT boundary --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_EXECUTION_TENSORBUFFER_H
#define TERA_EXECUTION_TENSORBUFFER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <random>
#include <vector>

namespace mlir::tera {

/// A tensor held the way the JIT expects it: contiguous row-major storage plus
/// the ranked memref descriptor that points at it.
class TensorBuffer {
public:
  explicit TensorBuffer(RankedTensorType type);

  /// Allocates a buffer with dynamic dimensions of \p type supplied by \p
  /// shape.
  TensorBuffer(RankedTensorType type, ArrayRef<int64_t> shape);

  TensorBuffer(const TensorBuffer &) = delete;
  TensorBuffer &operator=(const TensorBuffer &) = delete;
  TensorBuffer(TensorBuffer &&) = default;
  TensorBuffer &operator=(TensorBuffer &&) = default;

  /// Creates a result descriptor whose storage will be allocated by the callee.
  static TensorBuffer forResult(RankedTensorType type);

  /// Updates the tensor shape from the dimensions in the returned descriptor.
  void adoptDescriptorShape();

  /// Rejects unsupported JIT element types, identifying \p what in \p os.
  static LogicalResult checkElementType(Type elementType, StringRef what,
                                        raw_ostream &os = llvm::errs());

  /// Builds a ranked memref descriptor over caller-owned row-major storage.
  /// The allocated and aligned pointers both refer to \p data, with zero
  /// offset.
  static SmallVector<uint64_t> describe(void *data, ArrayRef<int64_t> shape);

  RankedTensorType getType() const { return type; }
  int64_t getNumElements() const { return type.getNumElements(); }
  unsigned getElementByteSize() const;

  /// Returns the mutable ranked memref descriptor used by the JIT.
  MutableArrayRef<uint64_t> getDescriptor() { return descriptor; }

  /// Fills the buffer with random values in [-spread, spread].
  void fill(std::mt19937_64 &generator, double spread);

  /// Returns the element at the row-major linear index as a double.
  double getElement(int64_t index) const;
  void setElement(int64_t index, double value);

private:
  char *getData() const;

  RankedTensorType type;
  std::vector<uint64_t> storage;
  SmallVector<uint64_t> descriptor;
};

} // namespace mlir::tera

#endif // TERA_EXECUTION_TENSORBUFFER_H
