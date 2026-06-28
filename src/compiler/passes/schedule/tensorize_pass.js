import { PrimFuncPass } from '../tir_pass.js';
import { Schedule } from '../../schedule/schedule.js';

const WMMA_TILE = 16;
const HALF_DTYPES = new Set(['f16', 'bf16']);

function findMatmulBlock(primFunc) {
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
  if (blocks.length === 0) return null;
  let matmul = null;
  for (const b of blocks) {
    if (!b.name.includes('matmul')) return null;
    if (b.reads.length >= 2 && b.writes.length >= 1) matmul = b;
  }
  return matmul;
}

export function detectWmmaMatmul(primFunc) {
  const matmul = findMatmulBlock(primFunc);
  if (!matmul) return null;
  const A = matmul.reads[0].buffer, B = matmul.reads[1].buffer, C = matmul.writes[0].buffer;
  if (!HALF_DTYPES.has(A.dtype) || !HALF_DTYPES.has(B.dtype) || C.dtype !== 'f32') return null;
  if (A.shape.length !== 2 || B.shape.length !== 2 || C.shape.length !== 2) return null;
  const M = C.shape[0], N = C.shape[1], K = A.shape[1];
  if (![M, N, K].every(d => typeof d === 'number' && d > 0 && d % WMMA_TILE === 0)) return null;
  const names = new Set();
  for (const [, buf] of primFunc.bufferMap) names.add(buf.name);
  if (!names.has(A.name) || !names.has(B.name) || !names.has(C.name)) return null;
  return { M, N, K, a: A.name, b: B.name, c: C.name };
}

export class AutoTensorizePass extends PrimFuncPass {
  constructor(config) {
    super('AutoTensorizePass', 'scheduling');
    this.config = config;
    this.target = config.target;
  }

  run(pf, ctx) {
    if (pf.cublasInfo || pf._tensorIntrin) return;
    if (!this.target || !this.target.isGPU()) return;
    const info = detectWmmaMatmul(pf);
    if (!info) return;
    new Schedule(pf).tensorize('wmma_16x16x16_f16f16f32', info);
    if (ctx && ctx.trace && ctx.trace.explainsEnabled) {
      ctx.trace.explain('tensorize', pf.name, 'wmma_16x16x16_f16f16f32',
        `auto-tensorized ${info.M}x${info.N}x${info.K} f16 GEMM`, { target: this.target.name });
    }
  }
}
