import { getCodegenEntry } from './codegen_registry.js';
import type { CodegenEntry, CodegenMetadata } from './codegen_registry.js';
import type { BlockNode, NodeSlots, PrimFunc, TirNode } from '../compiler/ir/tensor/nodes.js';
import type { CompilerContext } from '../compiler/pipeline/compiler_context.js';
import type { TargetFeatures } from './target.js';

export type PureMatmulInfo = { M: number; N: number; K: number; aIdx: number; bIdx: number; cIdx: number };
export type BackendPipelineOptions = { matmulBackend?: string; context?: CompilerContext | null };

export function detectPureMatmul(primFunc: PrimFunc): PureMatmulInfo | null {
  const blocks: BlockNode[] = [];
  const stack: TirNode[] = [primFunc.body];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n) continue;
    const slots = n as unknown as NodeSlots;
    if (n.type === 'BlockNode') { blocks.push(n as BlockNode); stack.push(slots.body as TirNode); continue; }
    if (slots.body) stack.push(slots.body as TirNode);
    if (slots.stmts) for (const s of slots.stmts as TirNode[]) stack.push(s);
    if (slots.thenBody) stack.push(slots.thenBody as TirNode);
    if (slots.elseBody) stack.push(slots.elseBody as TirNode);
  }
  let matmul: BlockNode | null = null;
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
  const M = C.shape[0] as number, N = C.shape[1] as number, K = A.shape[1] as number;
  if (![M, N, K].every(d => typeof d === 'number')) return null;
  const names: string[] = [];
  for (const [, buf] of primFunc.bufferMap) names.push(buf.name);
  const aIdx = names.indexOf(A.name), bIdx = names.indexOf(B.name), cIdx = names.indexOf(C.name);
  if (aIdx < 0 || bIdx < 0 || cIdx < 0) return null;
  return { M, N, K, aIdx, bIdx, cIdx };
}

export class CompiledKernel {
  name: string;
  source: string;
  target: TargetFeatures;
  metadata: Partial<CodegenMetadata>;

  constructor(name: string, source: string, target: TargetFeatures, metadata: Partial<CodegenMetadata> = {}) {
    this.name = name;
    this.source = source;
    this.target = target;
    this.metadata = metadata;
  }
}

export class BackendPipeline {
  target: TargetFeatures;
  matmulBackend: string;
  context: CompilerContext | null;

  constructor(target: TargetFeatures, options: BackendPipelineOptions = {}) {
    this.target = target;
    this.matmulBackend = options.matmulBackend || 'native';
    this.context = options.context || null;
  }

  compile(primFunc: PrimFunc): CompiledKernel {
    const entry = ((this.context && this.context.getCodegenEntry(this.target.kind)) || getCodegenEntry(this.target.kind)) as CodegenEntry | null;
    if (!entry) throw new Error(`Unsupported target kind: ${this.target.kind}`);
    const { source, metadata } = entry.compile(primFunc, this.target, this);
    return new CompiledKernel(primFunc.name, source, this.target, metadata);
  }

  compileAll(primFuncs: readonly PrimFunc[]): CompiledKernel[] {
    return primFuncs.map(f => this.compile(f));
  }
}
