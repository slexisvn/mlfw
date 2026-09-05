//===- TeraDialect.td - Tera dialect -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_DIALECT
#define TERA_DIALECT

include "Tera/IR/TeraInterfaces.td"
include "mlir/IR/OpBase.td"

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

    let hasConstantMaterializer = 1;

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

        /// On a differentiated function, names the function computing its
        /// result together with the values its derivative reads back, so a
        /// caller wanting both pays for the forward pass once.
        static constexpr ::llvm::StringLiteral kFwdAttrName = "tera.fwd";

        /// On a differentiated function, names the function computing its
        /// result together with the direction that result moves in.
        static constexpr ::llvm::StringLiteral kJvpAttrName = "tera.jvp";

        /// On a differentiated function, names the function taking those
        /// saved values and a seed and answering with the gradients.
        static constexpr ::llvm::StringLiteral kBwdAttrName = "tera.bwd";

        /// On a derivative function, the argument positions of the original
        /// function that its results are gradients for, in result order.
        static constexpr ::llvm::StringLiteral kDiffArgsAttrName =
            "tera.diff_args";
    }];
}

class Tera_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tera_Dialect, mnemonic,
           traits # [Tera_LoweringOpInterface, Tera_JvpOpInterface]>;

#endif
