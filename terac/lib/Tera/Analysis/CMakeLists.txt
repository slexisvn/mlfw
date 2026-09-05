add_mlir_library(MLIRTeraAnalysis
        Activity.cpp
        BlockTiles.cpp
        VectorTiles.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/Analysis

        LINK_LIBS PUBLIC
        MLIRDialectUtils
        MLIRIR
        MLIRIndexingMapOpInterface
        MLIRLinalgDialect
        MLIRSCFDialect
        )

tera_check_layering(Analysis "Conversion|Transforms|Execution|CAPI")
