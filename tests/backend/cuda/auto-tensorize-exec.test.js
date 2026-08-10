import { describe, it, expect } from 'vitest';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, ForNode, VariableNode, IntImmNode, ForKind
} from '../../../src/compiler/ir/tensor/nodes.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { CUDATarget } from '../../../src/backend/target.js';
import { RuntimeModule } from '../../../src/runtime/runtime.js';
import { AutoTensorizePass } from '../../../src/compiler/passes/schedule/tensorize_pass.js';
import { f32ToF16 } from '../../../src/tensor/utils/half.js';
import { cudaDeps } from './cuda-setup.js';
import { FuncAttr } from '../../../src/compiler/ir/func_attrs.js';

function matmulFunc(M, N, K) {
  const A = new Buffer('A', [M, K], 'f16', 'global');
  const B = new Buffer('B', [K, N], 'f16', 'global');
  const C = new Buffer('C', [M, N], 'f32', 'global');
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const blk = new BlockNode('matmul_acc', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
  let nest = blk;
  for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  return new PrimFunc('matmul_auto', [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

function refMatmul(a, b, M, N, K) {
  const c = new Float32Array(M * N);
  for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) { let s = 0; for (let k = 0; k < K; k++) s += a[m * K + k] * b[k * N + n]; c[m * N + n] = s; }
  return c;
}

describe.skipIf(!cudaDeps)('AutoTensorizePass — real-GPU WMMA execution', () => {
  it('auto-detects + tensorizes an f16 GEMM and runs correctly on the GPU', async () => {
    const M = 32, N = 32, K = 32;
    const f = matmulFunc(M, N, K);
    new AutoTensorizePass({ target: CUDATarget() }).run(f, {});
    expect(f.hasAttr(FuncAttr.TENSOR_INTRIN)).toBe(true);

    const af = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.1) * 0.5);
    const bf = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.1) * 0.5);
    const A = new Uint16Array(M * K), B = new Uint16Array(K * N);
    for (let i = 0; i < af.length; i++) A[i] = f32ToF16(af[i]);
    for (let i = 0; i < bf.length; i++) B[i] = f32ToF16(bf[i]);
    const C = new Float32Array(M * N);

    const kernel = new BackendPipeline(CUDATarget()).compile(f);
    const rm = new RuntimeModule('m');
    rm.addCompiledKernel(kernel);
    await rm.runAsync(kernel.name, A, B, C);

    const ref = refMatmul(af, bf, M, N, K);
    let e = 0, sc = 0;
    for (let i = 0; i < M * N; i++) { e = Math.max(e, Math.abs(C[i] - ref[i])); sc = Math.max(sc, Math.abs(ref[i])); }
    expect(e / (sc + 1e-9)).toBeLessThan(5e-2);
  });
});
