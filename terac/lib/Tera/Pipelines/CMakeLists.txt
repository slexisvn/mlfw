add_mlir_library(MLIRTeraPipelines
        CudaTarget.cpp
        HostTarget.cpp
        Pipelines.cpp
        Targets.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Tera/Pipelines

        LINK_LIBS PUBLIC
        MLIRAffineToStandard
        MLIRArithToLLVM
        MLIRBufferizationPipelines
        MLIRBufferizationToMemRef
        MLIRBufferizationTransforms
        MLIRControlFlowToLLVM
        MLIRFuncToLLVM
        MLIRGPUPipelines
        MLIRGPUTransforms
        MLIRLLVMIRTransforms
        MLIRLinalgTransforms
        MLIRMathToLLVM
        MLIRMemRefToLLVM
        MLIRMemRefTransforms
        MLIRPass
        MLIRReconcileUnrealizedCasts
        MLIRSCFToControlFlow
        MLIRSCFToGPU
        MLIRTeraConversion
        MLIRTransforms
        MLIRUBToLLVM
        MLIRVectorToLLVM
        MLIRVectorToSCF
        MLIRVectorTransforms
        )

tera_check_layering(Pipelines "Execution|CAPI")
