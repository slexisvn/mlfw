//===- TeraOps.td - Tera dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The tensor level of the tera compiler. Ops here keep whole-tensor semantics
// because that is the level reverse-mode AD needs: a closed VJP rule exists for
// `dot` and `reduce`, but not for the loop nest they lower to. Everything below
// this level is upstream `linalg`.
//
// This file is the entry point tablegen reads. Op definitions live in the
// per-family files it includes.
//
//===----------------------------------------------------------------------===//

#ifndef TERA_OPS
#define TERA_OPS

include "Tera/IR/TeraOpsBase.td"
include "Tera/IR/TeraOpsConstant.td"
include "Tera/IR/TeraOpsElementwise.td"
include "Tera/IR/TeraOpsShape.td"
include "Tera/IR/TeraOpsIndexing.td"
include "Tera/IR/TeraOpsWindow.td"
include "Tera/IR/TeraOpsContraction.td"
include "Tera/IR/TeraOpsAutodiff.td"
include "Tera/IR/TeraOpsControlFlow.td"

#endif // TERA_OPS
