import { describe, it, expect } from 'vitest';
import { Buffer } from '../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, BufferStoreNode, ForNode, VariableNode, IntImmNode, ForKind
} from '../../src/compiler/ir/tensor/nodes.js';
import { BackendPipeline } from '../../src/backend/pipeline.js';
import { CUDATarget } from '../../src/backend/target.js';

describe('CUDA tensor-core (wmma) codegen', () => {
  it('emits a valid wmma fp16 GEMM kernel when func.gpuWmma is set (structural; GPU-verified separately)', () => {
    const M = 32, N = 32, K = 32;
    const A = new Buffer('A', [M, K], 'f16', 'global');
    const B = new Buffer('B', [K, N], 'f16', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
    const blk = new BlockNode('mm', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
      [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
    let nest = blk;
    for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
    const func = new PrimFunc('matmul_wmma', [], nest, new Map([['A', A], ['B', B], ['C', C]]));
    func.gpuWmma = { M, N, K, a: 'A', b: 'B', c: 'C' };

    const src = new BackendPipeline(CUDATarget()).compile(func).source;
    expect(src).toMatch(/fragment<accumulator, 16, 16, 16, float>/);
    expect(src).toMatch(/fragment<matrix_a, 16, 16, 16, half/);
    expect(src).toMatch(/load_matrix_sync/);
    expect(src).toMatch(/mma_sync\(cf, af, bf, cf\)/);
    expect(src).toMatch(/store_matrix_sync/);
    expect(src).toMatch(/__half\* A/);
    expect(src).toMatch(/float\* C/);
  });

  it('emits a cp.async double-buffered (software-pipelined) GEMM when func.gpuPipelined is set', () => {
    const M = 64, N = 64, K = 64;
    const A = new Buffer('A', [M, K], 'f32', 'global');
    const B = new Buffer('B', [K, N], 'f32', 'global');
    const C = new Buffer('C', [M, N], 'f32', 'global');
    const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
    const blk = new BlockNode('mm', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
      [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
    let nest = blk;
    for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
    const func = new PrimFunc('gemm_pipe', [], nest, new Map([['A', A], ['B', B], ['C', C]]));
    func.gpuPipelined = { M, N, K, a: 'A', b: 'B', c: 'C', tile: 16 };

    const src = new BackendPipeline(CUDATarget()).compile(func).source;
    expect(src).toMatch(/__shared__ float As\[2\]/);
    expect(src).toMatch(/__pipeline_memcpy_async/);
    expect(src).toMatch(/__pipeline_commit/);
    expect(src).toMatch(/__pipeline_wait_prior/);
  });
});
