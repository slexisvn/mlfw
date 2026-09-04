//===- TeraDialect.td - Tera dialect -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_DIALECT
#define TERA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tera dialect definition.
//===----------------------------------------------------------------------===//

def Tera_Dialect : Dialect {
    let name = "tera";
    let summary = "A tera out-of-tree MLIR dialect.";
    let description = [{
        The tensor level of the tera compiler: whole-tensor ops with closed
        semantics, sitting above `linalg`. Reverse-mode AD runs here, before
        lowering, because a closed VJP rule exists for `dot` and `reduce` but
        not for the loop nest they become.
    }];
    let cppNamespace = "::mlir::tera";

    let useDefaultAttributePrinterParser = 1;

    // A dialect that owns a `ConstantLike` op has to be able to build one from
    // an attribute, or a fold that reaches a constant has nowhere to put it and
    // is dropped without a word. `-sccp` proves a value constant through this
    // hook and no other.
    let hasConstantMaterializer = 1;

    // The three attributes below are discardable: they sit on `func.func`,
    // which knows nothing about them. Nothing checks a discardable attribute
    // unless its dialect does, so without this a misspelt `tera.differentiable`
    // is not an error, it is a function that silently gets no derivative.
    let hasOperationAttrVerify = 1;

    let extraClassDeclaration = [{
        void registerAttributes();

        /// Marks a `func.func` that `-tera-autodiff` derives a reverse-mode
        /// derivative for.
        static constexpr ::llvm::StringLiteral kDifferentiableAttrName =
            "tera.differentiable";

        /// On a differentiated function, names the function holding its
        /// derivative. `-tera-autodiff` writes it, and it is how a caller
        /// finds the derivative without reconstructing its name.
        static constexpr ::llvm::StringLiteral kVjpAttrName = "tera.vjp";

        /// On a derivative function, the argument positions of the original
        /// function that its results are gradients for, in result order.
        static constexpr ::llvm::StringLiteral kDiffArgsAttrName =
            "tera.diff_args";
    }];
}

//===----------------------------------------------------------------------===//
// Base tera operation definition.
//===----------------------------------------------------------------------===//

class Tera_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tera_Dialect, mnemonic, traits>;

#endif // TERA_DIALECT
