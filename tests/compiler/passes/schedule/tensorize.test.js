import { describe, it, expect } from 'vitest';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, ForNode, VariableNode, IntImmNode, ForKind
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { BackendPipeline } from '../../../../src/backend/pipeline.js';
import { CPUTarget, CUDATarget } from '../../../../src/backend/target.js';
import { AutoTensorizePass, detectWmmaMatmul } from '../../../../src/compiler/passes/schedule/tensorize_pass.js';
import { FuncAttr } from '../../../../src/compiler/ir/func_attrs.js';

function matmulFunc(name, M, N, K, aDt = 'f16', bDt = 'f16', cDt = 'f32') {
  const A = new Buffer('A', [M, K], aDt, 'global');
  const B = new Buffer('B', [K, N], bDt, 'global');
  const C = new Buffer('C', [M, N], cDt, 'global');
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const blk = new BlockNode('matmul_acc', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
  let nest = blk;
  for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  return new PrimFunc(name, [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

describe('AutoTensorizePass — automatic tensor-core matmul detection', () => {
  it('detects an f16xf16->f32 GEMM divisible by 16 and returns the intrinsic info', () => {
    const info = detectWmmaMatmul(matmulFunc('mm', 32, 32, 32));
    expect(info).toEqual({ M: 32, N: 32, K: 32, a: 'A', b: 'B', c: 'C' });
  });

  it('rejects f32 matmul (cuBLAS/native territory) and non-16-divisible shapes', () => {
    expect(detectWmmaMatmul(matmulFunc('mm', 32, 32, 32, 'f32', 'f32', 'f32'))).toBe(null);
    expect(detectWmmaMatmul(matmulFunc('mm', 20, 32, 32))).toBe(null);
    expect(detectWmmaMatmul(matmulFunc('mm', 32, 32, 32, 'f16', 'f16', 'f16'))).toBe(null);
  });

  it('applies tensorize on a GPU target and codegen emits a WMMA kernel', () => {
    const pf = matmulFunc('mm_auto', 32, 32, 32);
    new AutoTensorizePass({ target: CUDATarget() }).run(pf, {});
    expect(pf.getAttr(FuncAttr.TENSOR_INTRIN)).toEqual({ name: 'wmma_16x16x16_f16f16f32', info: { M: 32, N: 32, K: 32, a: 'A', b: 'B', c: 'C' } });

    const src = new BackendPipeline(CUDATarget()).compile(pf).source;
    expect(src).toMatch(/mma_sync\(cf, af, bf, cf\)/);
    expect(src).toMatch(/load_matrix_sync/);
  });

  it('does nothing on a CPU target', () => {
    const pf = matmulFunc('mm_cpu', 32, 32, 32);
    new AutoTensorizePass({ target: CPUTarget() }).run(pf, {});
    expect(pf.hasAttr(FuncAttr.TENSOR_INTRIN)).toBe(false);
  });

  it('skips a func already routed to cuBLAS', () => {
    const pf = matmulFunc('mm_cublas', 32, 32, 32);
    pf.setAttr(FuncAttr.CUBLAS_INFO, { M: 32, N: 32, K: 32 });
    new AutoTensorizePass({ target: CUDATarget() }).run(pf, {});
    expect(pf.hasAttr(FuncAttr.TENSOR_INTRIN)).toBe(false);
  });
});
