add_mlir_library(MLIRTeraExecution
        DeviceMemory.cpp
        JitInvoker.cpp
        TensorBuffer.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/Execution

        LINK_LIBS PUBLIC
        MLIRExecutionEngine
        MLIRIR
        MLIRPass
        MLIRTeraPipelines
        )
