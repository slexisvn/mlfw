set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(tera-gradcheck
  tera-gradcheck.cpp

  EXPORT_SYMBOLS
  )
llvm_update_compile_flags(tera-gradcheck)
mlir_target_link_libraries(tera-gradcheck PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  )
target_link_libraries(tera-gradcheck PRIVATE
  MLIRTera
  MLIRTeraConversion
  MLIRTeraExecution
  MLIRTeraTransforms
  )

mlir_check_all_link_libraries(tera-gradcheck)
