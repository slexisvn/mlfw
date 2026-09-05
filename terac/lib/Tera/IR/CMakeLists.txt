add_mlir_dialect_library(MLIRTera
        TeraDialect.cpp
        TeraEnums.cpp
        TeraAutodiff.cpp
        TeraOps.cpp
        TeraOpsConstant.cpp
        TeraOpsElementwise.cpp
        TeraOpsShape.cpp
        TeraOpsIndexing.cpp
        TeraOpsWindow.cpp
        TeraOpsContraction.cpp
        TeraOpsAutodiff.cpp
        TeraOpsControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/IR

        DEPENDS
        MLIRTeraOpsIncGen
        MLIRTeraInterfacesIncGen
        MLIRTeraEnumsIncGen

        LINK_LIBS PUBLIC
        MLIRArithDialect
        MLIRControlFlowInterfaces
        MLIRFunctionInterfaces
        MLIRIR
        MLIRDialectUtils
        MLIRInferTypeOpInterface
        MLIRSideEffectInterfaces
        MLIRTensorDialect
        MLIRTransformUtils
        )

tera_check_layering(IR "Analysis|Conversion|Transforms|Execution|CAPI")
