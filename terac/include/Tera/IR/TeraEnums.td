//===- TeraEnums.td - Tera dialect enums -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_ENUMS
#define TERA_ENUMS

include "Tera/IR/TeraDialect.td"
include "mlir/IR/EnumAttr.td"

def Tera_REDUCE_SUM : I32EnumCase<"Sum", 0, "sum">;
def Tera_REDUCE_PRODUCT : I32EnumCase<"Product", 1, "product">;
def Tera_REDUCE_MAXIMUM : I32EnumCase<"Maximum", 2, "maximum">;
def Tera_REDUCE_MINIMUM : I32EnumCase<"Minimum", 3, "minimum">;
def Tera_REDUCE_MEAN : I32EnumCase<"Mean", 4, "mean">;

def Tera_ReduceKind : I32Enum<"ReduceKind",
    "The combiner a tera.reduce applies over the reduced axes",
    [Tera_REDUCE_SUM, Tera_REDUCE_PRODUCT, Tera_REDUCE_MAXIMUM,
     Tera_REDUCE_MINIMUM, Tera_REDUCE_MEAN]> {
  let cppNamespace = "::mlir::tera";
}

def Tera_ReduceKindAttr
    : EnumAttr<Tera_Dialect, Tera_ReduceKind, "reduce_kind">;

def Tera_POOL_MAX : I32EnumCase<"Max", 0, "max">;
def Tera_POOL_AVERAGE : I32EnumCase<"Average", 1, "average">;

def Tera_PoolKind : I32Enum<"PoolKind",
    "The reduction a tera.pool2d applies over each window",
    [Tera_POOL_MAX, Tera_POOL_AVERAGE]> {
  let cppNamespace = "::mlir::tera";
}

def Tera_PoolKindAttr : EnumAttr<Tera_Dialect, Tera_PoolKind, "pool_kind">;

def Tera_COMPARE_EQ : I32EnumCase<"Eq", 0, "eq">;
def Tera_COMPARE_NE : I32EnumCase<"Ne", 1, "ne">;
def Tera_COMPARE_LT : I32EnumCase<"Lt", 2, "lt">;
def Tera_COMPARE_LE : I32EnumCase<"Le", 3, "le">;
def Tera_COMPARE_GT : I32EnumCase<"Gt", 4, "gt">;
def Tera_COMPARE_GE : I32EnumCase<"Ge", 5, "ge">;

def Tera_ComparisonDirection : I32Enum<"ComparisonDirection",
    "The predicate a tera.compare applies elementwise",
    [Tera_COMPARE_EQ, Tera_COMPARE_NE, Tera_COMPARE_LT, Tera_COMPARE_LE,
     Tera_COMPARE_GT, Tera_COMPARE_GE]> {
  let cppNamespace = "::mlir::tera";
}

def Tera_ComparisonDirectionAttr
    : EnumAttr<Tera_Dialect, Tera_ComparisonDirection, "comparison_direction">;

#endif
