//===- TeraOps.td - Tera dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#endif
