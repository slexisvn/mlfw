set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name TeraConversion)
add_public_tablegen_target(MLIRTeraConversionPassesIncGen)

add_mlir_doc(Passes TeraConversionPasses Tera/ -gen-pass-doc)
