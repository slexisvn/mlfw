import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { DominatorFusionPass } from '../../../src/compiler/passes/fusion/dominator_fusion.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';

const F32 = ScalarType.F32;

function run(func, config = {}) {
  return new DominatorFusionPass(config).run(func);
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

function countReductionsInGroup(fusionOp) {
  const body = fusionOp.regions[0].entryBlock;
  let count = 0;
  for (const op of body.ops()) {
    if (op.opName === 'reduce') count++;
  }
  return count;
}

describe('DominatorFusionPass — reduction limit enforcement', () => {
  it('fuses elementwise chain (no reductions)', () => {
    const t = new TensorType([64, 64], F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    const fusions = findOps(func, 'fusion');
    expect(fusions.length).toBe(1);
  });

  it('fuses elementwise -> reduction (single reduction allowed)', () => {
    const t = new TensorType([8, 64], F32);
    const outT = new TensorType([8], F32);
    const func = buildFunction('f', [t, t], [outT], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const zero = b.scalarConstant(0.0, F32);
      b.returnOp([b.reduce(sum.getResult(0), zero.getResult(0), [1], 'sum').getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    const fusions = findOps(func, 'fusion');
    expect(fusions.length).toBeGreaterThanOrEqual(1);
    for (const f of fusions) {
      expect(countReductionsInGroup(f)).toBeLessThanOrEqual(1);
    }
  });

  it('does not fuse two reductions into the same group', () => {
    const t = new TensorType([4, 16], F32);
    const midT = new TensorType([4], F32);
    const func = buildFunction('f', [t], [midT], (b, args) => {
      const zero = b.scalarConstant(0.0, F32);
      const sum1 = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      const neg = b.neg(sum1.getResult(0));
      const expanded = b.broadcast(neg.getResult(0), [4, 16], [0]);
      const sum2 = b.reduce(expanded.getResult(0), zero.getResult(0), [1], 'sum');
      b.returnOp([sum2.getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    for (const f of fusions) {
      expect(countReductionsInGroup(f)).toBeLessThanOrEqual(1);
    }
  });
});

describe('DominatorFusionPass — no NaN/Inf in compiled output', () => {
  function compileWithDominator(func, target) {
    return compileGraph(func, target, { fusion: { dominator: true } });
  }

  function hasComputedNanInf(src) {
    const lines = src.split('\n');
    for (const line of lines) {
      const stripped = line.replace(/\/\/.*$/, '').replace(/;;.*$/, '').trim();
      if (!stripped) continue;
      if (/=\s*-?Infinity\b/.test(stripped)) continue;
      if (/\.const\s+-?Infinity\b/.test(stripped)) continue;
      if (/\bNaN\b/.test(stripped)) return true;
    }
    return false;
  }

  it('softmax with dominator fusion produces no NaN on CPU', () => {
    const t = new TensorType([4, 32], F32);
    const func = buildFunction('sm', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    const result = compileWithDominator(func, CPUTarget());
    for (const k of result.listKernels()) {
      expect(hasComputedNanInf(result.getSource(k))).toBe(false);
    }
  });

  it('softmax with dominator fusion produces no Inf on WASM', () => {
    const t = new TensorType([4, 32], F32);
    const func = buildFunction('sm_wasm', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    const result = compileWithDominator(func, WasmTarget());
    for (const k of result.listKernels()) {
      expect(hasComputedNanInf(result.getSource(k))).toBe(false);
    }
  });

  it('reduce+elementwise+reduce chain with dominator does not produce NaN', () => {
    const t = new TensorType([8, 16], F32);
    const midT = new TensorType([8], F32);
    const func = buildFunction('re_chain', [t], [midT], (b, args) => {
      const zero = b.scalarConstant(0.0, F32);
      const s1 = b.reduce(args[0], zero.getResult(0), [1], 'sum');
      const neg = b.neg(s1.getResult(0));
      const expanded = b.broadcast(neg.getResult(0), [8, 16], [0]);
      const s2 = b.reduce(expanded.getResult(0), zero.getResult(0), [1], 'sum');
      b.returnOp([s2.getResult(0)]);
    });

    const result = compileWithDominator(func, CPUTarget());
    for (const k of result.listKernels()) {
      expect(hasComputedNanInf(result.getSource(k))).toBe(false);
    }
  });
});

describe('DominatorFusionPass — maxReductions config', () => {
  it('respects maxReductions=1 (default)', () => {
    const t = new TensorType([64, 64], F32);
    const outT = new TensorType([64], F32);
    const func = buildFunction('f', [t, t], [outT], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const zero = b.scalarConstant(0.0, F32);
      b.returnOp([b.reduce(sum.getResult(0), zero.getResult(0), [1], 'sum').getResult(0)]);
    });

    run(func, { maxReductions: 1 });
    const fusions = findOps(func, 'fusion');
    for (const f of fusions) {
      expect(countReductionsInGroup(f)).toBeLessThanOrEqual(1);
    }
  });

  it('skips opaque ops (library ops)', () => {
    const t = new TensorType([64, 64], F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    const before = findOps(func, 'add').length + findOps(func, 'neg').length;
    run(func, { libraryOps: new Set(['add']) });
    const after = findOps(func, 'add').length + findOps(func, 'neg').length;

    expect(after).toBe(before);
  });
});
