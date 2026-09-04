# A shared library rather than another source in the Execution component,
# because what loads it is a process that never linked MLIR: it has to carry
# the compiler with it and expose exactly the handful of symbols in the header.
# The link components are the ones a driver sets, for the same reason -- the JIT
# emits code for the host, and that code generator is not pulled in by anything
# the dialect itself needs.
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_mlir_library(tera-capi
        SHARED
        Execution.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/CAPI
        )
llvm_update_compile_flags(tera-capi)
target_compile_definitions(tera-capi PRIVATE TERA_CAPI_BUILDING)
mlir_target_link_libraries(tera-capi PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRRegisterAllDialects
        MLIRRegisterAllExtensions
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRToLLVMIRTranslationRegistration
        )
target_link_libraries(tera-capi PRIVATE
        MLIRTera
        MLIRTeraConversion
        MLIRTeraExecution
        )
