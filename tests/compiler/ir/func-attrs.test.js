import { describe, it, expect } from 'vitest';
import { PrimFunc, SeqNode, BlockNode, BufferStoreNode, ForNode, VariableNode, IntImmNode, ForKind } from '../../../src/compiler/ir/tensor/nodes.js';
import { LIRFunc } from '../../../src/compiler/ir/lir/nodes.js';
import { FuncAttr } from '../../../src/compiler/ir/func_attrs.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { lowerToLIR } from '../../../src/compiler/passes/lowering/tensor_to_lir.js';
import { clonePrimFunc } from '../../../src/compiler/autotune/tune_ir.js';
import { CUDATarget } from '../../../src/backend/target.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { AutoTensorizePass } from '../../../src/compiler/passes/schedule/tensorize_pass.js';

function matmulFunc(M, N, K, aDt = 'f16', bDt = 'f16', cDt = 'f32') {
  const A = new Buffer('A', [M, K], aDt, 'global');
  const B = new Buffer('B', [K, N], bDt, 'global');
  const C = new Buffer('C', [M, N], cDt, 'global');
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const blk = new BlockNode('matmul_acc', [{ iterVar: i, binding: i }, { iterVar: j, binding: j }],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }], new BufferStoreNode(C, [i, j], new IntImmNode(0)));
  let nest = blk;
  for (const [v, e] of [[j, N], [i, M]]) nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  return new PrimFunc('mm', [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

describe('func attrs are first-class on PrimFunc and LIRFunc', () => {
  it('starts empty and round-trips values', () => {
    const pf = new PrimFunc('f', [], new SeqNode([]));
    expect(pf.hasAttr(FuncAttr.CUBLAS_INFO)).toBe(false);
    expect(pf.getAttr(FuncAttr.CUBLAS_INFO)).toBeNull();
    pf.setAttr(FuncAttr.CUBLAS_INFO, { M: 4 });
    expect(pf.getAttr(FuncAttr.CUBLAS_INFO)).toEqual({ M: 4 });
    expect(pf.removeAttr(FuncAttr.CUBLAS_INFO)).toBe(true);
    expect(pf.hasAttr(FuncAttr.CUBLAS_INFO)).toBe(false);
  });

  it('honours an explicit fallback', () => {
    const pf = new PrimFunc('f', [], new SeqNode([]));
    expect(pf.getAttr('missing', 'fallback')).toBe('fallback');
  });

  it('is available on LIRFunc too', () => {
    const lf = new LIRFunc('f', [], new SeqNode([]), new Map(), [], new Map(), null);
    lf.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
    expect(lf.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)).toBe(true);
  });

  it('does not share attr maps between instances', () => {
    const a = new PrimFunc('a', [], new SeqNode([]));
    const b = new PrimFunc('b', [], new SeqNode([]));
    a.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
    expect(b.hasAttr(FuncAttr.GPU_REGISTER_BLOCKED)).toBe(false);
  });
});

describe('attrs survive every IR boundary', () => {
  it('lowerToLIR carries attrs from PrimFunc to LIRFunc', () => {
    const pf = matmulFunc(32, 32, 32);
    pf.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);
    pf.setAttr(FuncAttr.CUBLAS_INFO, { M: 32, N: 32, K: 32 });

    const lir = lowerToLIR(pf, CUDATarget());

    expect(lir.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)).toBe(true);
    expect(lir.getAttr(FuncAttr.CUBLAS_INFO)).toEqual({ M: 32, N: 32, K: 32 });
  });

  it('clonePrimFunc copies attrs without aliasing the source map', () => {
    const pf = matmulFunc(32, 32, 32);
    pf.setAttr(FuncAttr.GPU_REGISTER_BLOCKED, true);

    const copy = clonePrimFunc(pf);
    expect(copy.getAttr(FuncAttr.GPU_REGISTER_BLOCKED)).toBe(true);

    copy.setAttr(FuncAttr.CONV_INFO, { k: 3 });
    expect(pf.hasAttr(FuncAttr.CONV_INFO)).toBe(false);
  });

  it('a tensorized PrimFunc still emits the intrinsic after LIR lowering', () => {
    const target = CUDATarget();
    const pf = matmulFunc(32, 32, 32);
    new AutoTensorizePass({ target }).run(pf, {});
    expect(pf.hasAttr(FuncAttr.TENSOR_INTRIN)).toBe(true);

    const lir = lowerToLIR(pf, target);
    const source = new BackendPipeline(target).compile(lir).source;

    expect(source).toMatch(/mma_sync/);
    expect(source).toMatch(/load_matrix_sync/);
  });
});
