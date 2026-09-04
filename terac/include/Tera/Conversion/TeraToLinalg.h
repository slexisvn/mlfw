//===- TeraToLinalg.h - tera to linalg conversion ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TERA_CONVERSION_TERATOLINALG_H
#define TERA_CONVERSION_TERATOLINALG_H

namespace mlir {
class RewritePatternSet;

namespace tera {
void populateTeraToLinalgPatterns(RewritePatternSet &patterns);

}
}

#endif
