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

    // An op answering for its own extents builds `tensor.dim` and the
    // arithmetic around it, so both dialects are loaded wherever tera is
    // rather than wherever someone remembered to ask for them.
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::tensor::TensorDialect",
    ];

    let useDefaultAttributePrinterParser = 1;

    let hasConstantMaterializer = 1;

    let hasOperationAttrVerify = 1;

    let hasRegionArgAttrVerify = 1;

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

        /// On a `func.func` argument, says the pointer the caller passes
        /// already points at memory on the device, so the kernels are handed
        /// it as it stands rather than a copy of it being staged each call.
        /// A weight is what this is for: uploaded once, read every step.
        static constexpr ::llvm::StringLiteral kDeviceResidentAttrName =
            "tera.device_resident";

        /// On a lowered op, the name of the tera op it came from, which is
        /// what a schedule addresses it by. `-convert-tera-to-linalg` writes
        /// it as `<function>.<op>.<n>`: readable, so a tuned schedule can be
        /// read and overridden by hand, and stable under everything but a
        /// change to the program, so that one keeps applying.
        static constexpr ::llvm::StringLiteral kScheduleAttrName =
            "tera.schedule";
    }];
}

class Tera_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tera_Dialect, mnemonic,
           traits # [Tera_LoweringOpInterface, Tera_JvpOpInterface]>;

#endif
