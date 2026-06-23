
import { createCPULibrarySelector, createGPULibrarySelector } from './library_selector.js';
import { getCodegenEntry, getSnippetBuilder } from './codegen_registry.js';

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
    const builder = getSnippetBuilder(this.metadata.kind);
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
    const entry = getCodegenEntry(this.target.kind);
    if (!entry) throw new Error(`Unsupported target kind: ${this.target.kind}`);
    const { source, metadata } = entry.compile(primFunc, this.target, this);
    return new CompiledKernel(primFunc.name, source, this.target, metadata);
  }

  compileAll(primFuncs) {
    return primFuncs.map(f => this.compile(f));
  }
}
