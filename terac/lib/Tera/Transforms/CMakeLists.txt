add_mlir_dialect_library(MLIRTeraTransforms
        Autodiff.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/Transforms

        DEPENDS
        MLIRTeraPassesIncGen

        LINK_LIBS PUBLIC
        MLIRTera
        MLIRFuncDialect
        MLIRPass
        MLIRTransformUtils
        )
