import { describe, it, expect } from 'vitest';
import { Buffer } from '../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, ForNode, VariableNode, IntImmNode, ForKind
} from '../../src/compiler/ir/tensor/nodes.js';
import { BackendPipeline } from '../../src/backend/pipeline.js';
import { CUDATarget } from '../../src/backend/target.js';
import { Schedule } from '../../src/compiler/schedule/schedule.js';
import { getCudaIntrin } from '../../src/backend/cuda/tensor_intrin.js';
import { FuncAttr } from '../../src/compiler/ir/func_attrs.js';

function matmulFunc(name, A, B, C, M, N) {
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const blk = new BlockNode('mm', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
  let nest = blk;
  for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  return new PrimFunc(name, [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

describe('CUDA tensor-intrinsic (tensorize) codegen', () => {
  it('tensorize primitive records the intrinsic on the func and the registry resolves it', () => {
    const M = 32, N = 32, K = 32;
    const A = new Buffer('A', [M, K], 'f16', 'global');
    const B = new Buffer('B', [K, N], 'f16', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const func = matmulFunc('mm_t', A, B, C, M, N);
    new Schedule(func).tensorize('wmma_16x16x16_f16f16f32', { M, N, K, a: 'A', b: 'B', c: 'C' });
    expect(func.getAttr(FuncAttr.TENSOR_INTRIN)).toEqual({ name: 'wmma_16x16x16_f16f16f32', info: { M, N, K, a: 'A', b: 'B', c: 'C' } });
    expect(typeof getCudaIntrin('wmma_16x16x16_f16f16f32')).toBe('function');
    expect(getCudaIntrin('no_such_intrin')).toBe(null);
  });

  it('emits a valid wmma fp16 GEMM kernel via the wmma intrinsic (structural; GPU-verified separately)', () => {
    const M = 32, N = 32, K = 32;
    const A = new Buffer('A', [M, K], 'f16', 'global');
    const B = new Buffer('B', [K, N], 'f16', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const func = matmulFunc('matmul_wmma', A, B, C, M, N);
    new Schedule(func).tensorize('wmma_16x16x16_f16f16f32', { M, N, K, a: 'A', b: 'B', c: 'C' });

    const src = new BackendPipeline(CUDATarget()).compile(func).source;
    expect(src).toMatch(/fragment<accumulator, 16, 16, 16, float>/);
    expect(src).toMatch(/fragment<matrix_a, 16, 16, 16, half/);
    expect(src).toMatch(/load_matrix_sync/);
    expect(src).toMatch(/mma_sync\(cf, af, bf, cf\)/);
    expect(src).toMatch(/store_matrix_sync/);
    expect(src).toMatch(/__half\* A/);
    expect(src).toMatch(/float\* C/);
  });

  it('emits a cp.async double-buffered (software-pipelined) GEMM via the pipelined intrinsic', () => {
    const M = 64, N = 64, K = 64;
    const A = new Buffer('A', [M, K], 'f32', 'global');
    const B = new Buffer('B', [K, N], 'f32', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const func = matmulFunc('gemm_pipe', A, B, C, M, N);
    new Schedule(func).tensorize('gemm_pipelined_f32', { M, N, K, a: 'A', b: 'B', c: 'C', tile: 16 });

    const src = new BackendPipeline(CUDATarget()).compile(func).source;
    expect(src).toMatch(/__shared__ float As\[2\]/);
    expect(src).toMatch(/__pipeline_memcpy_async/);
    expect(src).toMatch(/__pipeline_commit/);
    expect(src).toMatch(/__pipeline_wait_prior/);
  });

  it('throws on an unknown tensor intrinsic at codegen', () => {
    const M = 16, N = 16, K = 16;
    const A = new Buffer('A', [M, K], 'f16', 'global');
    const B = new Buffer('B', [K, N], 'f16', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const func = matmulFunc('mm_bad', A, B, C, M, N);
    func.setAttr(FuncAttr.TENSOR_INTRIN, { name: 'nonexistent_intrin', info: { M, N, K, a: 'A', b: 'B', c: 'C' } });
    expect(() => new BackendPipeline(CUDATarget()).compile(func)).toThrow(/unknown tensor intrinsic/);
  });
});
