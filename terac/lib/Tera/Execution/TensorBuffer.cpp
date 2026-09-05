//===- TensorBuffer.cpp - A tensor across the JIT boundary ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tera/Execution/TensorBuffer.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

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

/// Every float is carried the same way, through the semantics its type
/// declares, so a narrow one is not a case to add here: f16 and bf16 differ
/// from f32 only in what `APFloat` is told, and rounding a `double` into them
/// is the same call. What an element cannot be is a float wider than 64 bits,
/// which would not fit back through this interface, or an integer of a width
/// nothing on either side of the boundary uses.
static bool isCarriable(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return floatType.getWidth() <= 64;
  return elementType.isSignlessInteger(32) ||
         elementType.isSignlessInteger(64);
}

/// The raw bits of one element, read as the little-endian integer they are.
static uint64_t readBits(const char *slot, unsigned bytes) {
  uint64_t bits = 0;
  std::memcpy(&bits, slot, bytes);
  return bits;
}

static void writeBits(char *slot, unsigned bytes, uint64_t bits) {
  std::memcpy(slot, &bits, bytes);
}

LogicalResult TensorBuffer::checkElementType(Type elementType, StringRef what,
                                             raw_ostream &os) {
  if (isCarriable(elementType))
    return success();
  os << what << " has element type " << elementType
     << "; only a float of at most 64 bits, i32 and i64 cross the JIT "
        "boundary\n";
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
  Type elementType = type.getElementType();
  assert(isCarriable(elementType) && "checked when the buffer was created");
  unsigned bytes = getElementByteSize();
  const char *slot = getData() + index * bytes;

  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    APFloat held(floatType.getFloatSemantics(),
                 APInt(floatType.getWidth(), readBits(slot, bytes)));
    return held.convertToDouble();
  }
  if (elementType.isSignlessInteger(32))
    return *reinterpret_cast<const int32_t *>(slot);
  return static_cast<double>(*reinterpret_cast<const int64_t *>(slot));
}

void TensorBuffer::setElement(int64_t index, double value) {
  Type elementType = type.getElementType();
  assert(isCarriable(elementType) && "checked when the buffer was created");
  unsigned bytes = getElementByteSize();
  char *slot = getData() + index * bytes;

  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    APFloat rounded(value);
    bool lost = false;
    rounded.convert(floatType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &lost);
    writeBits(slot, bytes, rounded.bitcastToAPInt().getZExtValue());
    return;
  }
  if (elementType.isSignlessInteger(32))
    *reinterpret_cast<int32_t *>(slot) = static_cast<int32_t>(value);
  else
    *reinterpret_cast<int64_t *>(slot) = static_cast<int64_t>(value);
}
