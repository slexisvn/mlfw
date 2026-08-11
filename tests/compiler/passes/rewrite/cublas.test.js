import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../../src/backend/target.js';
import { CublasRewritePass } from '../../../../src/compiler/passes/rewrite/cublas_rewrite.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { T as t } from '../../../_utils/ir_fixture.js';


function buildMatmul() {
  return buildFunction('mm', [t([2, 3]), t([3, 4])], [t([2, 4])], (b, [a, w]) => {
    b.returnOp([b.matmul(a, w).getResult(0)]);
  });
}

describe('CublasRewritePass (XLA GemmRewriter-style: dot -> cublas_gemm)', () => {
  it('rewrites dot to cublas_gemm in the graph IR', () => {
    const func = buildMatmul();
    expect(func.opsArray().filter(o => o.opName === 'dot').length).toBe(1);
    const result = new CublasRewritePass().run(func);
    expect(result).toBe(PassResult.CHANGED);
    expect(func.opsArray().filter(o => o.opName === 'dot').length).toBe(0);
    expect(func.opsArray().filter(o => o.opName === 'cublas_gemm').length).toBe(1);
  });

  it('the rewritten cublas_gemm compiles and runs as a correct matmul (dot-equivalent lowering)', () => {
    const func = buildMatmul();
    new CublasRewritePass().run(func);
    const c = compileGraph(func, CPUTarget());
    const A = new Float32Array([1, 2, 3, 4, 5, 6]);
    const W = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    const o = new Float32Array(8);
    c.run(c.listKernels()[0], A, W, o);
    expect([...o]).toEqual([1, 2, 3, 0, 4, 5, 6, 0]);
  });

  it('does not rewrite an ineligible (batched) matmul — keeps cublas_gemm honest', () => {
    const func = buildFunction('bmm', [t([2, 3, 4]), t([2, 4, 5])], [t([2, 3, 5])], (b, [a, w]) => {
      b.returnOp([b.matmul(a, w).getResult(0)]);
    });
    expect(func.opsArray().filter(o => o.opName === 'dot').length).toBe(1);
    expect(new CublasRewritePass().run(func)).toBe(PassResult.UNCHANGED);
    expect(func.opsArray().filter(o => o.opName === 'cublas_gemm').length).toBe(0);
  });

  it('is a no-op when there are no dot ops', () => {
    const func = buildFunction('add', [t([4]), t([4])], [t([4])], (b, [a, x]) => {
      b.returnOp([b.add(a, x).getResult(0)]);
    });
    expect(new CublasRewritePass().run(func)).toBe(PassResult.UNCHANGED);
  });
});
