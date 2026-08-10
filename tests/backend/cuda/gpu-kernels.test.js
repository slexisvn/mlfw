import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../../src/backend/target.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, ForNode, VariableNode, IntImmNode, ForKind
} from '../../../src/compiler/ir/tensor/nodes.js';
import { f32ToF16 } from '../../../src/tensor/utils/half.js';
import { RuntimeModule } from '../../../src/runtime/runtime.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { cudaDeps } from './cuda-setup.js';
import { FuncAttr } from '../../../src/compiler/ir/func_attrs.js';

const F32 = ScalarType.F32;
const t = (s) => new TensorType(s, F32);
const numel = (s) => s.reduce((a, b) => a * b, 1);

function matmulFunc(M, N, K, aDt, bDt, cDt) {
  const A = new Buffer('A', [M, K], aDt, 'global');
  const B = new Buffer('B', [K, N], bDt, 'global');
  const C = new Buffer('C', [M, N], cDt, 'global');
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const blk = new BlockNode('mm', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
  let nest = blk;
  for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  return new PrimFunc('matmul', [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}
function refMatmul(a, b, M, N, K) {
  const c = new Float32Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) { let s = 0; for (let k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n]; c[m * N + n] = s; }
  return c;
}
async function runRawCuda(func, ...args) {
  const kernel = new BackendPipeline(CUDATarget()).compile(func);
  const rm = new RuntimeModule('m');
  rm.addCompiledKernel(kernel);
  await rm.runAsync(kernel.name, ...args);
}

describe.skipIf(!cudaDeps)('flash / wmma / cp.async / collectives on real GPU', () => {
  it('flash attention matches CPU', async () => {
    const B = 1, H = 2, L = 4, D = 8, sc = 1 / Math.sqrt(D), shp = [B, H, L, D];
    const mk = () => buildFunction('attn', [t(shp), t(shp), t(shp)], [t(shp)], (b, [q, k, v]) =>
      b.returnOp([b.scaledDotProductAttention(q, k, v, sc).getResult(0)]));
    const Q = Float32Array.from({ length: numel(shp) }, (_, n) => Math.sin(n * 0.53 + 1));
    const K = Float32Array.from({ length: numel(shp) }, (_, n) => Math.cos(n * 0.4 + 2));
    const V = Float32Array.from({ length: numel(shp) }, (_, n) => Math.sin(n * 0.3 + 3));
    const cpu = new Float32Array(numel(shp)); compileGraph(mk(), CPUTarget()).run('attn', Q, K, V, cpu);
    const gpu = new Float32Array(numel(shp)); await compileGraph(mk(), CUDATarget()).runAsync('attn', Q, K, V, gpu);
    for (let i = 0; i < gpu.length; i++) expect(gpu[i]).toBeCloseTo(cpu[i], 4);
  });

  it('KV-cache decode (Lq=1 over a cache) matches CPU', async () => {
    const B = 1, H = 2, Lq = 1, Lk = 6, D = 8, sc = 1 / Math.sqrt(D);
    const qs = [B, H, Lq, D], ks = [B, H, Lk, D];
    const mk = () => buildFunction('dec', [t(qs), t(ks), t(ks)], [t(qs)], (b, [q, k, v]) =>
      b.returnOp([b.scaledDotProductAttention(q, k, v, sc).getResult(0)]));
    const Q = Float32Array.from({ length: numel(qs) }, (_, n) => Math.sin(n * 0.5 + 1));
    const K = Float32Array.from({ length: numel(ks) }, (_, n) => Math.cos(n * 0.4 + 2));
    const V = Float32Array.from({ length: numel(ks) }, (_, n) => Math.sin(n * 0.3 + 3));
    const cpu = new Float32Array(numel(qs)); compileGraph(mk(), CPUTarget()).run('dec', Q, K, V, cpu);
    const gpu = new Float32Array(numel(qs)); await compileGraph(mk(), CUDATarget()).runAsync('dec', Q, K, V, gpu);
    for (let i = 0; i < gpu.length; i++) expect(gpu[i]).toBeCloseTo(cpu[i], 4);
  });

  it('all_reduce sums across devices on GPU', async () => {
    const mk = () => buildFunction('ar', [t([2, 3])], [t([2, 3])], (b, [x]) => b.returnOp([b.allReduce(x).getResult(0)]));
    const gpu = new Float32Array(6);
    await compileGraph(mk(), CUDATarget()).runAsync('ar', new Float32Array([0, 1, 2, 10, 11, 12]), gpu);
    expect([...gpu]).toEqual([10, 12, 14, 10, 12, 14]);
  });

  it('wmma fp16 tensor-core GEMM matches reference', async () => {
    const M = 32, N = 32, K = 32;
    const f = matmulFunc(M, N, K, 'f16', 'f16', 'f32');
    f.setAttr(FuncAttr.TENSOR_INTRIN, { name: 'wmma_16x16x16_f16f16f32', info: { M, N, K, a: 'A', b: 'B', c: 'C' } });
    const af = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.1) * 0.5);
    const bf = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.1) * 0.5);
    const A = new Uint16Array(M * K), B = new Uint16Array(K * N);
    for (let i = 0; i < af.length; i++) A[i] = f32ToF16(af[i]);
    for (let i = 0; i < bf.length; i++) B[i] = f32ToF16(bf[i]);
    const C = new Float32Array(M * N);
    await runRawCuda(f, A, B, C);
    const ref = refMatmul(af, bf, M, N, K);
    let e = 0, sc = 0;
    for (let i = 0; i < M * N; i++) { e = Math.max(e, Math.abs(C[i] - ref[i])); sc = Math.max(sc, Math.abs(ref[i])); }
    expect(e / (sc + 1e-9)).toBeLessThan(5e-2);
  });

  it('cp.async software-pipelined GEMM matches reference', async () => {
    const M = 64, N = 64, K = 64;
    const f = matmulFunc(M, N, K, 'f32', 'f32', 'f32');
    f.setAttr(FuncAttr.TENSOR_INTRIN, { name: 'gemm_pipelined_f32', info: { M, N, K, a: 'A', b: 'B', c: 'C', tile: 16 } });
    const A = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.05) * 0.5);
    const B = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.05) * 0.5);
    const C = new Float32Array(M * N);
    await runRawCuda(f, A, B, C);
    const ref = refMatmul(A, B, M, N, K);
    for (let i = 0; i < M * N; i++) expect(C[i]).toBeCloseTo(ref[i], 3);
  });
});
