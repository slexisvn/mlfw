import { TargetKind } from './target.js';
import { CPUCodegen } from './cpu/codegen.js';
import { CUDACodegen } from './cuda/codegen.js';
import { WasmCodegen } from './wasm/codegen.js';
import { WebGPUCodegen } from './webgpu/codegen.js';
import { buildSnippet as cpuSnippet } from './cpu/snippet.js';
import { buildSnippet as wasmSnippet } from './wasm/snippet.js';
import { buildSnippet as webgpuSnippet } from './webgpu/snippet.js';
import { buildSnippet as cudaSnippet } from './cuda/snippet.js';

const _byTargetKind = new Map();
const _snippetByRuntimeKind = new Map();

export function registerCodegen(targetKind, entry) {
  _byTargetKind.set(targetKind, entry);
  if (entry.runtimeKind && entry.snippet) _snippetByRuntimeKind.set(entry.runtimeKind, entry.snippet);
}

export function getCodegenEntry(targetKind) {
  return _byTargetKind.get(targetKind) || null;
}

export function getSnippetBuilder(runtimeKind) {
  return _snippetByRuntimeKind.get(runtimeKind) || null;
}

registerCodegen(TargetKind.CPU, {
  runtimeKind: 'js',
  snippet: cpuSnippet,
  compile(primFunc, target) {
    const source = new CPUCodegen(target).generate(primFunc);
    return { source, metadata: { kind: 'js', paramCount: primFunc.params.length } };
  },
});

registerCodegen(TargetKind.WASM, {
  runtimeKind: 'wasm',
  snippet: wasmSnippet,
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
  snippet: webgpuSnippet,
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
  snippet: cudaSnippet,
  compile(primFunc, target, pipeline) {
    if (pipeline && pipeline.matmulBackend === 'cublas' && primFunc.cublasInfo) {
      return {
        source: '',
        metadata: { kind: 'cuda', cublas: primFunc.cublasInfo, outputIndices: [primFunc.cublasInfo.cIdx] },
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
