//===- TensorBuffer.cpp - A tensor across the JIT boundary ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/TensorBuffer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tera;

TensorBuffer::TensorBuffer(RankedTensorType type, ArrayRef<int64_t> shape)
    : TensorBuffer(RankedTensorType::get(shape, type.getElementType())) {}

TensorBuffer::TensorBuffer(RankedTensorType type) : type(type) {
  size_t bytes = static_cast<size_t>(type.getNumElements()) *
                 (type.getElementTypeBitWidth() / 8);
  storage.assign((bytes + 7) / 8, 0);
  descriptor = describe(storage.data(), type.getShape());
}

SmallVector<uint64_t> TensorBuffer::describe(void *data,
                                             ArrayRef<int64_t> shape) {
  SmallVector<uint64_t> descriptor;
  descriptor.push_back(reinterpret_cast<uint64_t>(data));
  descriptor.push_back(reinterpret_cast<uint64_t>(data));
  descriptor.push_back(0);
  descriptor.append(shape.begin(), shape.end());
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0; --axis)
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  descriptor.append(strides.begin(), strides.end());
  return descriptor;
}

TensorBuffer TensorBuffer::forResult(RankedTensorType type) {
  SmallVector<int64_t> shape(type.getRank(), 0);
  TensorBuffer buffer(type.hasStaticShape()
                          ? type
                          : RankedTensorType::get(shape,
                                                  type.getElementType()));
  buffer.type = type;
  llvm::fill(buffer.descriptor, 0);
  return buffer;
}

void TensorBuffer::adoptDescriptorShape() {
  if (type.hasStaticShape())
    return;
  ArrayRef<uint64_t> sizes =
      ArrayRef<uint64_t>(descriptor).slice(3, type.getRank());
  SmallVector<int64_t> shape(sizes.begin(), sizes.end());
  type = RankedTensorType::get(shape, type.getElementType());
}

/// The element types a JIT-compiled function can be handed and read back.
/// Written once: `checkElementType` reports the set by asking for a kind, and
/// the accessors below switch over the closed enum rather than repeating the
/// membership test, so adding a type is one case in each switch and no new
/// list.
enum class ElementKind { F32, F64, I32, I64 };

static std::optional<ElementKind> classifyElementType(Type elementType) {
  if (elementType.isF32())
    return ElementKind::F32;
  if (elementType.isF64())
    return ElementKind::F64;
  if (elementType.isSignlessInteger(32))
    return ElementKind::I32;
  if (elementType.isSignlessInteger(64))
    return ElementKind::I64;
  return std::nullopt;
}

/// Requires an element type previously accepted by checkElementType.
static ElementKind checkedKind(Type elementType) {
  std::optional<ElementKind> kind = classifyElementType(elementType);
  assert(kind && "element type checked when the buffer was created");
  return *kind;
}

LogicalResult TensorBuffer::checkElementType(Type elementType, StringRef what,
                                             raw_ostream &os) {
  if (classifyElementType(elementType))
    return success();
  os << what << " has element type " << elementType
     << "; only f32, f64, i32 and i64 cross the JIT boundary\n";
  return failure();
}

unsigned TensorBuffer::getElementByteSize() const {
  return type.getElementTypeBitWidth() / 8;
}

char *TensorBuffer::getData() const {
  auto *aligned = reinterpret_cast<char *>(descriptor[1]);
  return aligned + descriptor[2] * getElementByteSize();
}

void TensorBuffer::fill(std::mt19937_64 &generator, double spread) {
  std::uniform_real_distribution<double> draw(-spread, spread);
  for (int64_t index = 0; index < getNumElements(); ++index)
    setElement(index, draw(generator));
}

double TensorBuffer::getElement(int64_t index) const {
  const char *slot = getData() + index * getElementByteSize();
  switch (checkedKind(type.getElementType())) {
  case ElementKind::F32:
    return *reinterpret_cast<const float *>(slot);
  case ElementKind::F64:
    return *reinterpret_cast<const double *>(slot);
  case ElementKind::I32:
    return *reinterpret_cast<const int32_t *>(slot);
  case ElementKind::I64:
    return static_cast<double>(*reinterpret_cast<const int64_t *>(slot));
  }
  llvm_unreachable("every element kind is handled");
}

void TensorBuffer::setElement(int64_t index, double value) {
  char *slot = getData() + index * getElementByteSize();
  switch (checkedKind(type.getElementType())) {
  case ElementKind::F32:
    *reinterpret_cast<float *>(slot) = static_cast<float>(value);
    return;
  case ElementKind::F64:
    *reinterpret_cast<double *>(slot) = value;
    return;
  case ElementKind::I32:
    *reinterpret_cast<int32_t *>(slot) = static_cast<int32_t>(value);
    return;
  case ElementKind::I64:
    *reinterpret_cast<int64_t *>(slot) = static_cast<int64_t>(value);
    return;
  }
  llvm_unreachable("every element kind is handled");
}
