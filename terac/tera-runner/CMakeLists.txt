set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(tera-runner
  tera-runner.cpp

  EXPORT_SYMBOLS
  )
llvm_update_compile_flags(tera-runner)
mlir_target_link_libraries(tera-runner PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  )
target_link_libraries(tera-runner PRIVATE
  MLIRTera
  MLIRTeraConversion
  MLIRTeraExecution
  )

mlir_check_all_link_libraries(tera-runner)
