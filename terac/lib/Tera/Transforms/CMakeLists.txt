add_mlir_dialect_library(MLIRTeraTransforms
        Autodiff.cpp
        ForwardMode.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/Transforms

        DEPENDS
        MLIRTeraPassesIncGen

        LINK_LIBS PUBLIC
        MLIRTera
        MLIRTeraAnalysis
        MLIRFuncDialect
        MLIRPass
        MLIRTransformUtils
        )

tera_check_layering(Transforms "Conversion|Execution|CAPI")
