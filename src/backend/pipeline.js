import { TargetKind } from './target.js';
import { CPUCodegen } from './cpu/codegen.js';
import { GPUCodegen } from './gpu/codegen.js';
import { WasmCodegen } from './wasm/codegen.js';
import { WebGPUCodegen } from './webgpu/codegen.js';
import { createCPULibrarySelector, createGPULibrarySelector } from './library_selector.js';
import { buildSnippet as cpuSnippet } from './cpu/snippet.js';
import { buildSnippet as wasmSnippet } from './wasm/snippet.js';
import { buildSnippet as webgpuSnippet } from './webgpu/snippet.js';
import { buildSnippet as gpuSnippet } from './gpu/snippet.js';

const SNIPPET_BUILDERS = { js: cpuSnippet, wasm: wasmSnippet, webgpu: webgpuSnippet, cuda: gpuSnippet };

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
  constructor(target) {
    this.target = target;
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
    if (this.target.isGPU()) return this._compileGPU(primFunc);
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

  _compileGPU(primFunc) {
    const codegen = new GPUCodegen(this.target);
    const kernel = codegen.generate(primFunc);
    return new CompiledKernel(primFunc.name, kernel.source, this.target, {
      kind: 'cuda',
      blockDim: kernel.blockDim,
      gridDim: kernel.gridDim,
      sharedMemBytes: kernel.sharedMemBytes,
      params: kernel.params
    });
  }
}
