add_mlir_dialect_library(MLIRTeraConversion
        ApplySchedule.cpp
        AttachWorkgroupMemory.cpp
        StageGpuBuffers.cpp
        TeraToLinalg.cpp
        TileAndFuse.cpp
        TileContractionToShared.cpp
        TileParallelLoops.cpp
        VectorizeLinalg.cpp
        VerifyGpuMapping.cpp
        TeraToLinalgConstant.cpp
        TeraToLinalgElementwise.cpp
        TeraToLinalgShape.cpp
        TeraToLinalgIndexing.cpp
        TeraToLinalgWindow.cpp
        TeraToLinalgContraction.cpp
        TeraToLinalgAutodiff.cpp
        TeraToLinalgControlFlow.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/Conversion

        DEPENDS
        MLIRTeraConversionPassesIncGen

        LINK_LIBS PUBLIC
        MLIRAffineDialect
        MLIRArithDialect
        MLIRArithUtils
        MLIRBufferizationDialect
        MLIRDialectUtils
        MLIRFuncDialect
        MLIRGPUDialect
        MLIRGPUTransforms
        MLIRIndexingMapOpInterface
        MLIRLinalgDialect
        MLIRLinalgTransforms
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRPass
        MLIRSCFDialect
        MLIRSCFTransforms
        MLIRSCFUtils
        MLIRTensorDialect
        MLIRTera
        MLIRTeraAnalysis
        MLIRTilingInterface
        MLIRTransformDialect
        MLIRTransformDialectTransforms
        MLIRTransforms
        MLIRVectorDialect
        )

tera_check_layering(Conversion "Pipelines|Execution|CAPI")
