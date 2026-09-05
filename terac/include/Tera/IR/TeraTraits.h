//===- TeraTraits.h - Tera op traits ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_IR_TERATRAITS_H
#define TERA_IR_TERATRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::tera {
template <typename ConcreteType>
class AcceptsDynamicShapes
    : public OpTrait::TraitBase<ConcreteType, AcceptsDynamicShapes> {};

template <typename ConcreteType>
class LinearOperands : public OpTrait::TraitBase<ConcreteType, LinearOperands> {
};

template <typename ConcreteType>
class MultilinearOperands
    : public OpTrait::TraitBase<ConcreteType, MultilinearOperands> {};

}

#endif
