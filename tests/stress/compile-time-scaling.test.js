import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/backend/target.js';
import { F32 } from '../_utils/ir_fixture.js';

const OPS = ['exp', 'tanh', 'sin', 'sigmoid', 'cos'];

function chain(n) {
  const t = new TensorType([256], F32);
  return buildFunction('c', [t], [t], (b, a) => {
    let v = a[0];
    for (let i = 0; i < n; i++) v = b[OPS[i % OPS.length]](v).getResult(0);
    b.returnOp([v]);
  });
}

function timeCompile(n) {
  const times = [];
  for (let r = 0; r < 3; r++) {
    const t0 = performance.now();
    compileGraph(chain(n), CPUTarget());
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[1];
}

describe('compile-time scaling (sub-quadratic guard)', () => {
  it('doubling op count grows compile time sub-quadratically', () => {
    timeCompile(128);

    const sizes = [256, 512, 1024, 2048];
    const t = sizes.map(timeCompile);

    for (let i = 1; i < sizes.length; i++) {
      const ratio = t[i] / Math.max(t[i - 1], 0.05);
      expect(ratio,
        `compile time ${sizes[i - 1]}->${sizes[i]} ops grew ${ratio.toFixed(2)}x (expected <3x; quadratic would be ~4x)`
      ).toBeLessThan(3.0);
    }

    const endToEnd = t[t.length - 1] / Math.max(t[0], 0.05);
    expect(endToEnd,
      `256->2048 ops (8x) grew ${endToEnd.toFixed(1)}x (expected <~16x; quadratic would be ~64x)`
    ).toBeLessThan(16);
  });
});
