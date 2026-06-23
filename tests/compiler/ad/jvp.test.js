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
