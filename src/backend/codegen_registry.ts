import { TargetKind } from './target.js';
import { CPUCodegen } from './cpu/codegen.js';
import { CUDACodegen } from './cuda/codegen.js';
import { WasmCodegen } from './wasm/codegen.js';
import { WebGPUCodegen } from './webgpu/codegen.js';
import { FuncAttr } from '../compiler/ir/func_attrs.js';
import type { PrimFunc } from '../compiler/ir/tensor/nodes.js';
import type { TargetFeatures } from './target.js';
import type { BackendPipeline } from './pipeline.js';

export type CublasInfo = { M: number; N: number; K: number; transB: boolean; aIdx: number; bIdx: number; cIdx: number };
export type CodegenMetadata = Record<string, unknown> & { kind: string };
export type CodegenOutput = { source: string; metadata: CodegenMetadata };
export type CodegenEntry = {
  runtimeKind: string;
  compile(primFunc: PrimFunc, target: TargetFeatures, pipeline?: BackendPipeline): CodegenOutput;
};

const _byTargetKind = new Map<string, CodegenEntry>();

export function registerCodegen(targetKind: string, entry: CodegenEntry): void {
  _byTargetKind.set(targetKind, entry);
}

export function getCodegenEntry(targetKind: string): CodegenEntry | null {
  return _byTargetKind.get(targetKind) || null;
}

registerCodegen(TargetKind.CPU, {
  runtimeKind: 'js',
  compile(primFunc: PrimFunc, target: TargetFeatures): CodegenOutput {
    const source = new CPUCodegen(target).generate(primFunc);
    return { source, metadata: { kind: 'js', paramCount: primFunc.params.length } };
  },
});

registerCodegen(TargetKind.WASM, {
  runtimeKind: 'wasm',
  compile(primFunc: PrimFunc, target: TargetFeatures): CodegenOutput {
    const result = new WasmCodegen(target).generate(primFunc);
    const metadata: CodegenMetadata = {
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
  compile(primFunc: PrimFunc, target: TargetFeatures): CodegenOutput {
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
  compile(primFunc: PrimFunc, target: TargetFeatures, pipeline?: BackendPipeline): CodegenOutput {
    const cublasInfo = primFunc.getAttr ? primFunc.getAttr<CublasInfo>(FuncAttr.CUBLAS_INFO) : null;
    if (pipeline && pipeline.matmulBackend === 'cublas' && cublasInfo) {
      return {
        source: '',
        metadata: { kind: 'cuda', cublas: cublasInfo, outputIndices: [(cublasInfo as CublasInfo).cIdx] },
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
