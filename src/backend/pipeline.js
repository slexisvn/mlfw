
import { CPUCodegen } from './cpu/codegen.js';
import { CUDACodegen } from './cuda/codegen.js';
import { WasmCodegen } from './wasm/codegen.js';
import { WebGPUCodegen } from './webgpu/codegen.js';
import { createCPULibrarySelector, createGPULibrarySelector } from './library_selector.js';
import { buildSnippet as cpuSnippet } from './cpu/snippet.js';
import { buildSnippet as wasmSnippet } from './wasm/snippet.js';
import { buildSnippet as webgpuSnippet } from './webgpu/snippet.js';
import { buildSnippet as cudaSnippet } from './cuda/snippet.js';

const SNIPPET_BUILDERS = { js: cpuSnippet, wasm: wasmSnippet, webgpu: webgpuSnippet, cuda: cudaSnippet };

export function detectPureMatmul(primFunc) {
  const blocks = [];
  const stack = [primFunc.body];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    if (n.type === 'BlockNode') { blocks.push(n); stack.push(n.body); continue; }
    if (n.body) stack.push(n.body);
    if (n.stmts) for (const s of n.stmts) stack.push(s);
    if (n.thenBody) stack.push(n.thenBody);
    if (n.elseBody) stack.push(n.elseBody);
  }
  let matmul = null;
  for (const b of blocks) {
    if (b.name.includes('matmul')) {
      if (b.reads.length >= 2 && b.writes.length >= 1) matmul = b;
    } else {
      return null;
    }
  }
  if (!matmul) return null;
  const A = matmul.reads[0].buffer, B = matmul.reads[1].buffer, C = matmul.writes[0].buffer;
  if (A.dtype !== 'f32' || B.dtype !== 'f32' || C.dtype !== 'f32') return null;
  if (A.shape.length !== 2 || B.shape.length !== 2 || C.shape.length !== 2) return null;
  const M = C.shape[0], N = C.shape[1], K = A.shape[1];
  if (![M, N, K].every(d => typeof d === 'number')) return null;
  const names = [];
  for (const [, buf] of primFunc.bufferMap) names.push(buf.name);
  const aIdx = names.indexOf(A.name), bIdx = names.indexOf(B.name), cIdx = names.indexOf(C.name);
  if (aIdx < 0 || bIdx < 0 || cIdx < 0) return null;
  return { M, N, K, aIdx, bIdx, cIdx };
}

export class CompiledKernel {
  constructor(name, source, target, metadata = {}) {
    this.name = name;
    this.source = source;
    this.target = target;
    this.metadata = metadata;
  }

  snippet() {
    const builder = SNIPPET_BUILDERS[this.metadata.kind];
    if (!builder) throw new Error(`No snippet for kind: ${this.metadata.kind}`);
    return builder(this);
  }
}

export class BackendPipeline {
  constructor(target, options = {}) {
    this.target = target;
    this.matmulBackend = options.matmulBackend || 'native';
    this.librarySelector = target.isCPU()
      ? createCPULibrarySelector(target)
      : target.isGPU()
        ? createGPULibrarySelector(target)
        : null;
  }

  compile(primFunc) {
    if (this.target.isWasm()) return this._compileWasm(primFunc);
    if (this.target.isCPU()) return this._compileCPU(primFunc);
    if (this.target.isWebGPU()) return this._compileWebGPU(primFunc);
    if (this.target.isGPU()) return this._compileCUDA(primFunc);
    throw new Error(`Unsupported target kind: ${this.target.kind}`);
  }

  compileAll(primFuncs) {
    return primFuncs.map(f => this.compile(f));
  }

  _compileCPU(primFunc) {
    const codegen = new CPUCodegen(this.target);
    const source = codegen.generate(primFunc);
    return new CompiledKernel(primFunc.name, source, this.target, {
      kind: 'js',
      paramCount: primFunc.params.length
    });
  }

  _compileWasm(primFunc) {
    const codegen = new WasmCodegen(this.target);
    const result = codegen.generate(primFunc);
    const meta = {
      kind: 'wasm',
      memoryPages: result.memoryPages,
      bufferOffsets: result.bufferOffsets,
      imports: result.imports,
      params: result.params,
      bufferMap: primFunc.bufferMap,
    };
    if (result.parallel) {
      meta.parallel = result.parallel;
    }
    return new CompiledKernel(primFunc.name, result.wat, this.target, meta);
  }

  _compileWebGPU(primFunc) {
    const codegen = new WebGPUCodegen(this.target);
    const kernel = codegen.generate(primFunc);
    return new CompiledKernel(primFunc.name, kernel.source, this.target, {
      kind: 'webgpu',
      workgroupSize: kernel.workgroupSize,
      dispatchSize: kernel.dispatchSize,
      sharedMemBytes: kernel.sharedMemBytes,
      params: kernel.params,
      bindings: kernel.bindings
    });
  }

  _compileCUDA(primFunc) {
    if (this.matmulBackend === 'cublas' && primFunc.cublasInfo) {
      return new CompiledKernel(primFunc.name, '', this.target, {
        kind: 'cuda',
        cublas: primFunc.cublasInfo,
        outputIndices: [primFunc.cublasInfo.cIdx]
      });
    }
    const codegen = new CUDACodegen(this.target);
    const kernel = codegen.generate(primFunc);
    return new CompiledKernel(primFunc.name, kernel.source, this.target, {
      kind: 'cuda',
      blockDim: kernel.blockDim,
      gridDim: kernel.gridDim,
      sharedMemBytes: kernel.sharedMemBytes,
      params: kernel.params,
      outputIndices: kernel.outputIndices,
      scratch: kernel.scratch
    });
  }
}
