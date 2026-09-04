set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls -name Tera)
add_public_tablegen_target(MLIRTeraPassesIncGen)

add_mlir_doc(Passes TeraPasses Tera/ -gen-pass-doc)
