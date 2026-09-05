set(LIBS
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRTera
    MLIRTeraConversion
    MLIRTeraPipelines
    MLIRTeraTransforms
    MLIRToLLVMIRTranslationRegistration
  )
add_llvm_executable(tera-opt tera-opt.cpp)

llvm_update_compile_flags(tera-opt)
target_link_libraries(tera-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tera-opt)

install(TARGETS tera-opt
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT tera-opt)
