import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { buildForwardDiff, getJVPRule, registerJVPRule } from '../../../src/compiler/ad/jvp.js';

const t = (s) => new TensorType(s, ScalarType.F32);

describe('forward-mode AD (JVP)', () => {
  it('buildForwardDiff produces a jvp(args, tangents) -> output-tangents function that runs correctly', () => {
    const fwd = buildFunction('f', [t([2]), t([2])], [t([2])], (b, [a, bb]) => {
      b.returnOp([b.mul(a, bb).getResult(0)]);
    });

    const jvp = buildForwardDiff(fwd);
    const c = compileGraph(jvp, CPUTarget());

    const A = new Float32Array([2, 3]);
    const B = new Float32Array([4, 5]);
    const dA = new Float32Array([1, 0]);
    const dB = new Float32Array([0, 1]);
    const out = new Float32Array(2);

    c.run(c.listKernels()[0], A, B, dA, dB, out);

    expect([...out]).toEqual([4, 3]);
  });

  it('forward-diffs division (y = a/b => dy = (da - y*db)/b)', () => {
    const fwd = buildFunction('f', [t([2]), t([2])], [t([2])], (b, [a, bb]) => {
      b.returnOp([b.div(a, bb).getResult(0)]);
    });
    const c = compileGraph(buildForwardDiff(fwd), CPUTarget());
    const A = new Float32Array([6, 8]);
    const B = new Float32Array([2, 4]);
    const dA = new Float32Array([1, 0]);
    const dB = new Float32Array([0, 1]);
    const out = new Float32Array(2);
    c.run(c.listKernels()[0], A, B, dA, dB, out);
    expect([...out]).toEqual([0.5, -0.5]);
  });

  it('forward-diffs an add+mul chain (y = (a+b)*a => dy = (da+db)*a + (a+b)*da)', () => {
    const fwd = buildFunction('f', [t([2]), t([2])], [t([2])], (b, [a, bb]) => {
      const s = b.add(a, bb).getResult(0);
      b.returnOp([b.mul(s, a).getResult(0)]);
    });
    const jvp = buildForwardDiff(fwd);
    const c = compileGraph(jvp, CPUTarget());
    const A = new Float32Array([2, 1]);
    const B = new Float32Array([3, 5]);
    const dA = new Float32Array([1, 0]);
    const dB = new Float32Array([0, 1]);
    const out = new Float32Array(2);
    c.run(c.listKernels()[0], A, B, dA, dB, out);
    expect([...out]).toEqual([7, 1]);
  });
});

describe('forward-mode AD (JVP) — composite/linear/elementwise rules', () => {
  it('dot is bilinear: dy = da·b + a·db', () => {
    const fwd = buildFunction('f', [t([3]), t([3])], [t([])], (b, [a, bb]) => {
      b.returnOp([b.dot(a, bb, [0], [0]).getResult(0)]);
    });
    const c = compileGraph(buildForwardDiff(fwd), CPUTarget());
    const out = new Float32Array(1);
    c.run(c.listKernels()[0], new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6]),
      new Float32Array([1, 0, 0]), new Float32Array([0, 0, 1]), out);
    expect(out[0]).toBeCloseTo(4 + 3, 5);
  });

  it('reduce-sum tangent is the sum of tangents', () => {
    const fwd = buildFunction('f', [t([3])], [t([])], (b, [x]) => {
      const init = b.scalarConstant(0, 'f32').getResult(0);
      b.returnOp([b.reduce(x, init, [0], 'sum').getResult(0)]);
    });
    const c = compileGraph(buildForwardDiff(fwd), CPUTarget());
    const out = new Float32Array(1);
    c.run(c.listKernels()[0], new Float32Array([5, 6, 7]), new Float32Array([1, 1, 1]), out);
    expect(out[0]).toBeCloseTo(3, 5);
  });

  it('maximum (relu) routes the tangent through the larger operand', () => {
    const fwd = buildFunction('f', [t([2])], [t([2])], (b, [x]) => {
      b.returnOp([b.relu(x).getResult(0)]);
    });
    const c = compileGraph(buildForwardDiff(fwd), CPUTarget());
    const out = new Float32Array(2);
    c.run(c.listKernels()[0], new Float32Array([-1, 2]), new Float32Array([1, 1]), out);
    expect([...out]).toEqual([0, 1]);
  });

  it('sigmoid tangent is y(1-y)dx', () => {
    const fwd = buildFunction('f', [t([1])], [t([1])], (b, [x]) => {
      b.returnOp([b.sigmoid(x).getResult(0)]);
    });
    const c = compileGraph(buildForwardDiff(fwd), CPUTarget());
    const out = new Float32Array(1);
    c.run(c.listKernels()[0], new Float32Array([0]), new Float32Array([1]), out);
    expect(out[0]).toBeCloseTo(0.25, 5);
  });

  it('transpose tangent is the transposed tangent (linear)', () => {
    const fwd = buildFunction('f', [t([2, 3])], [t([3, 2])], (b, [x]) => {
      b.returnOp([b.transpose(x, [1, 0]).getResult(0)]);
    });
    const c = compileGraph(buildForwardDiff(fwd), CPUTarget());
    const x = new Float32Array([0, 0, 0, 0, 0, 0]);
    const dx = new Float32Array([1, 2, 3, 4, 5, 6]);
    const out = new Float32Array(6);
    c.run(c.listKernels()[0], x, dx, out);
    expect([...out]).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it('throws on an op without a JVP rule (strict, no silent zero)', () => {
    const fwd = buildFunction('f', [t([2, 2])], [t([2, 2])], (b, [x]) => {
      b.returnOp([b.layernorm(x, x, x).getResult(0)]);
    });
    expect(() => buildForwardDiff(fwd)).toThrow(/no JVP rule/);
  });
});
