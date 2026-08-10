import { TargetKind } from './target.js';
import { CPUCodegen } from './cpu/codegen.js';
import { CUDACodegen } from './cuda/codegen.js';
import { WasmCodegen } from './wasm/codegen.js';
import { WebGPUCodegen } from './webgpu/codegen.js';
import { FuncAttr } from '../compiler/ir/func_attrs.js';

const _byTargetKind = new Map();

export function registerCodegen(targetKind, entry) {
  _byTargetKind.set(targetKind, entry);
}

export function getCodegenEntry(targetKind) {
  return _byTargetKind.get(targetKind) || null;
}

registerCodegen(TargetKind.CPU, {
  runtimeKind: 'js',
  compile(primFunc, target) {
    const source = new CPUCodegen(target).generate(primFunc);
    return { source, metadata: { kind: 'js', paramCount: primFunc.params.length } };
  },
});

registerCodegen(TargetKind.WASM, {
  runtimeKind: 'wasm',
  compile(primFunc, target) {
    const result = new WasmCodegen(target).generate(primFunc);
    const metadata = {
      kind: 'wasm',
      memoryPages: result.memoryPages,
      bufferOffsets: result.bufferOffsets,
      imports: result.imports,
      params: result.params,
      bufferMap: primFunc.bufferMap,
    };
    if (result.parallel) metadata.parallel = result.parallel;
    return { source: result.wat, metadata };
  },
});

registerCodegen(TargetKind.WEBGPU, {
  runtimeKind: 'webgpu',
  compile(primFunc, target) {
    const kernel = new WebGPUCodegen(target).generate(primFunc);
    return {
      source: kernel.source,
      metadata: {
        kind: 'webgpu',
        workgroupSize: kernel.workgroupSize,
        dispatchSize: kernel.dispatchSize,
        sharedMemBytes: kernel.sharedMemBytes,
        params: kernel.params,
        bindings: kernel.bindings,
      },
    };
  },
});

registerCodegen(TargetKind.CUDA, {
  runtimeKind: 'cuda',
  compile(primFunc, target, pipeline) {
    const cublasInfo = primFunc.getAttr ? primFunc.getAttr(FuncAttr.CUBLAS_INFO) : null;
    if (pipeline && pipeline.matmulBackend === 'cublas' && cublasInfo) {
      return {
        source: '',
        metadata: { kind: 'cuda', cublas: cublasInfo, outputIndices: [cublasInfo.cIdx] },
      };
    }
    const kernel = new CUDACodegen(target).generate(primFunc);
    return {
      source: kernel.source,
      metadata: {
        kind: 'cuda',
        blockDim: kernel.blockDim,
        gridDim: kernel.gridDim,
        sharedMemBytes: kernel.sharedMemBytes,
        params: kernel.params,
        outputIndices: kernel.outputIndices,
        scratch: kernel.scratch,
      },
    };
  },
});
