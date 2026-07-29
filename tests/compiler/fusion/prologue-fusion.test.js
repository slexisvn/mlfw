import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { EpilogueFusionPass } from '../../../src/compiler/passes/fusion/epilogue_fusion.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';

const F32 = ScalarType.F32;
const I32 = ScalarType.I32;

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) if (op.opName === opName) result.push(op);
  return result;
}

describe('prologue fusion — convert folded into dot', () => {
  it('folds a single-use convert on the lhs into fused_dot_epilogue with a prologue cast', () => {
    const xt = new TensorType([4, 6], I32);
    const yt = new TensorType([6, 5], F32);
    const outt = new TensorType([4, 5], F32);
    const func = buildFunction('pro', [xt, yt], [outt], (b, args) => {
      const xf = b.convert(args[0], 'f32');
      b.returnOp([b.matmul(xf.getResult(0), args[1]).getResult(0)]);
    });

    const result = new EpilogueFusionPass({ target: { enableEpilogueFusion: true } }).run(func);
    expect(result).toBe(PassResult.CHANGED);

    expect(findOps(func, 'convert').length).toBe(0);
    expect(findOps(func, 'dot').length).toBe(0);
    const fused = findOps(func, 'fused_dot_epilogue');
    expect(fused.length).toBe(1);
    expect(fused[0].getAttr('lhs_prologue_cast')).toBe('f32');
    expect(fused[0].getAttr('rhs_prologue_cast')).toBe(undefined);
  });

  it('does not fold a convert that has other uses', () => {
    const xt = new TensorType([4, 6], I32);
    const yt = new TensorType([6, 5], F32);
    const outt = new TensorType([4, 5], F32);
    const func = buildFunction('pro2', [xt, yt], [outt, new TensorType([4, 6], F32)], (b, args) => {
      const xf = b.convert(args[0], 'f32');
      const mm = b.matmul(xf.getResult(0), args[1]);
      b.returnOp([mm.getResult(0), xf.getResult(0)]);
    });

    new EpilogueFusionPass({ target: { enableEpilogueFusion: true } }).run(func);
    expect(findOps(func, 'convert').length).toBe(1);
  });

  it('compiles and matches a reference matmul (CPU, prologue active)', () => {
    const xt = new TensorType([4, 6], I32);
    const yt = new TensorType([6, 5], F32);
    const outt = new TensorType([4, 5], F32);
    const func = buildFunction('pro3', [xt, yt], [outt], (b, args) => {
      const xf = b.convert(args[0], 'f32');
      b.returnOp([b.matmul(xf.getResult(0), args[1]).getResult(0)]);
    });

    const res = compileGraph(func, CPUTarget({ enableEpilogueFusion: true }));

    const x = new Int32Array(24);
    for (let i = 0; i < 24; i++) x[i] = (i % 5) - 2;
    const y = new Float32Array(30);
    for (let i = 0; i < 30; i++) y[i] = (i % 7) * 0.25 - 0.5;
    const out = new Float32Array(20);
    res.run('pro3', x, y, out);

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 5; j++) {
        let ref = 0;
        for (let k = 0; k < 6; k++) ref += x[i * 6 + k] * y[k * 5 + j];
        expect(Math.abs(out[i * 5 + j] - ref)).toBeLessThan(1e-4);
      }
    }
  });
});
