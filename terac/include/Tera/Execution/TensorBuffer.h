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
class TensorBuffer {
public:
  explicit TensorBuffer(RankedTensorType type);

  TensorBuffer(RankedTensorType type, ArrayRef<int64_t> shape);

  TensorBuffer(const TensorBuffer &) = delete;
  TensorBuffer &operator=(const TensorBuffer &) = delete;
  TensorBuffer(TensorBuffer &&) = default;
  TensorBuffer &operator=(TensorBuffer &&) = default;

  static TensorBuffer forResult(RankedTensorType type);

  void adoptDescriptorShape();

  static LogicalResult checkElementType(Type elementType, StringRef what,
                                        raw_ostream &os = llvm::errs());

  static SmallVector<uint64_t> describe(void *data, ArrayRef<int64_t> shape);

  RankedTensorType getType() const { return type; }
  int64_t getNumElements() const { return type.getNumElements(); }
  unsigned getElementByteSize() const;

  MutableArrayRef<uint64_t> getDescriptor() { return descriptor; }

  void fill(std::mt19937_64 &generator, double spread);

  double getElement(int64_t index) const;
  void setElement(int64_t index, double value);

private:
  char *getData() const;

  RankedTensorType type;
  std::vector<uint64_t> storage;
  SmallVector<uint64_t> descriptor;
};

}

#endif
