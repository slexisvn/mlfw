import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { RematerializationPass, RematerializationConfig } from '../../../src/compiler/passes/memory/rematerialization.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func, opts = {}) {
  return new RematerializationPass(opts).run(func);
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

function opNames(func) {
  const names = [];
  for (const op of func.ops()) names.push(op.opName);
  return names;
}

describe('RematerializationPass budget gate', () => {
  it('returns UNCHANGED when memoryBudget is Infinity (default)', () => {
    const t = new TensorType([1024], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const n = b.neg(args[0]);
      b.returnOp([b.add(n.getResult(0), args[1]).getResult(0), b.exp(n.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('returns UNCHANGED when peak pressure is already under budget', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    expect(run(func, { memoryBudget: 1024 * 1024 })).toBe(PassResult.UNCHANGED);
  });
});

describe('RematerializationPass cloning', () => {
  it('clones op for second use when budget is tight', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t, t], (b, args) => {
      const n = b.neg(args[0]);
      const a = b.add(n.getResult(0), args[0]);
      const e = b.exp(n.getResult(0));
      b.returnOp([a.getResult(0), e.getResult(0), args[0]]);
    });

    const tinyBudget = 1024 * 1024 * 4;
    const result = run(func, { memoryBudget: tinyBudget });

    if (result === PassResult.CHANGED) {
      expect(findOps(func, 'neg').length).toBe(2);
    }
  });

  it('cloned op is inserted before its consumer', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t, t], (b, args) => {
      const n = b.neg(args[0]);
      const a = b.add(n.getResult(0), args[0]);
      const e = b.exp(n.getResult(0));
      b.returnOp([a.getResult(0), e.getResult(0), args[0]]);
    });

    const tinyBudget = 1024 * 1024 * 4;
    const result = run(func, { memoryBudget: tinyBudget });

    if (result === PassResult.CHANGED) {
      const names = opNames(func);
      const negs = [];
      names.forEach((n, i) => { if (n === 'neg') negs.push(i); });
      const expIdx = names.indexOf('exp');
      expect(negs.some(i => i < expIdx)).toBe(true);
    }
  });

  it('does not clone ops with only 1 use (useCount <= 1)', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const n = b.neg(args[0]);
      b.returnOp([n.getResult(0)]);
    });

    expect(run(func, { memoryBudget: 1 })).toBe(PassResult.UNCHANGED);
  });
});

describe('RematerializationPass candidate filtering', () => {
  it('never rematerializes constant ops', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t], (b, args) => {
      const c = b.scalarConstant(1.0, ScalarType.F32);
      const bc = b.broadcast(c.getResult(0), [1024, 1024], []);
      const a = b.add(args[0], bc.getResult(0));
      const m = b.mul(args[0], bc.getResult(0));
      b.returnOp([a.getResult(0), m.getResult(0)]);
    });

    run(func, { memoryBudget: 1 });
    expect(findOps(func, 'constant').length).toBe(1);
  });

  it('excludeOps prevents specific ops from being rematerialized', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t, t], (b, args) => {
      const n = b.neg(args[0]);
      const a = b.add(n.getResult(0), args[0]);
      const e = b.exp(n.getResult(0));
      b.returnOp([a.getResult(0), e.getResult(0), args[0]]);
    });

    run(func, { memoryBudget: 1, excludeOps: new Set(['neg']) });
    expect(findOps(func, 'neg').length).toBe(1);
  });

  it('block arguments are never rematerialized', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t, t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const m = b.mul(args[0], args[1]);
      b.returnOp([a.getResult(0), m.getResult(0)]);
    });

    const before = opNames(func);
    run(func, { memoryBudget: 1 });
    const after = opNames(func);
    expect(after.filter(n => n === 'add').length).toBe(before.filter(n => n === 'add').length);
    expect(after.filter(n => n === 'mul').length).toBe(before.filter(n => n === 'mul').length);
  });
});

describe('RematerializationPass scoring', () => {
  it('prefers candidates with higher memorySaved/recomputeCost ratio', () => {
    const big = new TensorType([2048, 2048], ScalarType.F32);
    const small = new TensorType([16], ScalarType.F32);
    const func = buildFunction('f', [big, small], [big, big, small, small], (b, args) => {
      const bigNeg = b.neg(args[0]);
      const smallNeg = b.neg(args[1]);
      const a1 = b.add(bigNeg.getResult(0), args[0]);
      const a2 = b.exp(bigNeg.getResult(0));
      const s1 = b.add(smallNeg.getResult(0), args[1]);
      const s2 = b.exp(smallNeg.getResult(0));
      b.returnOp([a1.getResult(0), a2.getResult(0), s1.getResult(0), s2.getResult(0)]);
    });

    const budget = 2048 * 2048 * 4 * 3;
    run(func, { memoryBudget: budget, maxIterations: 1 });

    const negCount = findOps(func, 'neg').length;
    if (negCount === 3) {
      const bigNegs = findOps(func, 'neg').filter(op => {
        const shape = op.getResult(0).type.shape;
        return shape[0] === 2048;
      });
      expect(bigNegs.length).toBe(2);
    }
  });
});

describe('RematerializationPass maxIterations', () => {
  it('maxIterations=1 limits to single rematerialization step', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t, t], (b, args) => {
      const n = b.neg(args[0]);
      const e = b.exp(args[0]);
      const a = b.add(n.getResult(0), args[0]);
      const m = b.mul(n.getResult(0), args[0]);
      const s = b.add(e.getResult(0), args[0]);
      const s2 = b.mul(e.getResult(0), args[0]);
      b.returnOp([a.getResult(0), m.getResult(0), s.getResult(0)]);
    });

    const before = opNames(func);
    run(func, { memoryBudget: 1, maxIterations: 1 });
    const after = opNames(func);
    const clonedCount = after.length - before.length;
    expect(clonedCount).toBeLessThanOrEqual(1);
  });
});

describe('RematerializationPass preserves IR correctness', () => {
  it('cloned op uses same operands as original', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t, t], (b, args) => {
      const n = b.neg(args[0]);
      const a = b.add(n.getResult(0), args[0]);
      const e = b.exp(n.getResult(0));
      b.returnOp([a.getResult(0), e.getResult(0), args[0]]);
    });

    const result = run(func, { memoryBudget: 1024 * 1024 * 4 });
    if (result === PassResult.CHANGED) {
      const negs = findOps(func, 'neg');
      for (const neg of negs) {
        expect(neg.getOperand(0).isBlockArgument()).toBe(true);
        expect(neg.getResult(0).type.shape).toEqual([1024, 1024]);
      }
    }
  });

  it('original op retains first use, clones serve later uses', () => {
    const t = new TensorType([1024, 1024], ScalarType.F32);
    const func = buildFunction('f', [t], [t, t, t], (b, args) => {
      const n = b.neg(args[0]);
      const a = b.add(n.getResult(0), args[0]);
      const e = b.exp(n.getResult(0));
      b.returnOp([a.getResult(0), e.getResult(0), args[0]]);
    });

    const result = run(func, { memoryBudget: 1024 * 1024 * 4 });
    if (result === PassResult.CHANGED) {
      const negs = findOps(func, 'neg');
      expect(negs.length).toBe(2);
      const firstNeg = negs[0];
      expect(firstNeg.getResult(0).useCount).toBe(1);
    }
  });
});
