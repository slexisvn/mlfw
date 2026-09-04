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

/// Carried by an op whose lowering can materialise its destination from
/// extents it can reach, so a `?` in its type is a width the pattern reads
/// rather than a size it would have to guess.
///
/// The trait is opt-in for the same reason the rule it replaces was a list:
/// an op added to the dialect carries no `?` until someone states that its
/// pattern can size a dynamic destination. Stating it on the op keeps the
/// answer beside the op's own semantics instead of in the pass that asks.
template <typename ConcreteType>
class AcceptsDynamicShapes
    : public OpTrait::TraitBase<ConcreteType, AcceptsDynamicShapes> {};

} // namespace mlir::tera

#endif // TERA_IR_TERATRAITS_H
