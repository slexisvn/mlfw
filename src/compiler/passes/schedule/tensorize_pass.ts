import { PrimFuncPass } from '../tir_pass.js';
import { Schedule } from '../../schedule/schedule.js';
import { FuncAttr } from '../../ir/func_attrs.js';
import type { BlockNode, NodeSlots, PrimFunc, TirNode } from '../../ir/tensor/nodes.js';
import type { TirPassCtx } from '../tir_pass.js';
import type { CompilerConfig, CompileTarget } from '../../pipeline/pipeline_types.js';

export type WmmaMatmulInfo = { M: number; N: number; K: number; a: string; b: string; c: string };

const WMMA_TILE = 16;
const HALF_DTYPES = new Set(['f16', 'bf16']);

function findMatmulBlock(primFunc: PrimFunc): BlockNode | null {
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
  if (blocks.length === 0) return null;
  let matmul: BlockNode | null = null;
  for (const b of blocks) {
    if (!b.name.includes('matmul')) return null;
    if (b.reads.length >= 2 && b.writes.length >= 1) matmul = b;
  }
  return matmul;
}

export function detectWmmaMatmul(primFunc: PrimFunc): WmmaMatmulInfo | null {
  const matmul = findMatmulBlock(primFunc);
  if (!matmul) return null;
  const A = matmul.reads[0].buffer, B = matmul.reads[1].buffer, C = matmul.writes[0].buffer;
  if (!HALF_DTYPES.has(A.dtype) || !HALF_DTYPES.has(B.dtype) || C.dtype !== 'f32') return null;
  if (A.shape.length !== 2 || B.shape.length !== 2 || C.shape.length !== 2) return null;
  const M = C.shape[0] as number, N = C.shape[1] as number, K = A.shape[1] as number;
  if (![M, N, K].every(d => typeof d === 'number' && d > 0 && d % WMMA_TILE === 0)) return null;
  const names = new Set<string>();
  for (const [, buf] of primFunc.bufferMap) names.add(buf.name);
  if (!names.has(A.name) || !names.has(B.name) || !names.has(C.name)) return null;
  return { M, N, K, a: A.name, b: B.name, c: C.name };
}

export class AutoTensorizePass extends PrimFuncPass {
  config: CompilerConfig;
  target: CompileTarget | null;

  constructor(config: CompilerConfig) {
    super('AutoTensorizePass', 'scheduling');
    this.config = config;
    this.target = config.target;
  }

  override run(pf: PrimFunc, ctx: TirPassCtx): void {
    if (pf.hasAttr(FuncAttr.CUBLAS_INFO) || pf.hasAttr(FuncAttr.TENSOR_INTRIN)) return;
    if (!this.target || !this.target.isGPU()) return;
    const info = detectWmmaMatmul(pf);
    if (!info) return;
    new Schedule(pf).tensorize('wmma_16x16x16_f16f16f32', info as never);
    if (ctx && ctx.trace && ctx.trace.explainsEnabled) {
      ctx.trace.explain('tensorize', pf.name, 'wmma_16x16x16_f16f16f32',
        `auto-tensorized ${info.M}x${info.N}x${info.K} f16 GEMM`, { target: (this.target as CompileTarget).name });
    }
  }
}
