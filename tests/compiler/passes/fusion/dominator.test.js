import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../../src/compiler/ir/graph/types.js';
import { DominatorFusionPass } from '../../../../src/compiler/passes/fusion/dominator_fusion.js';
import { PassResult } from '../../../../src/compiler/passes/pass.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../../src/backend/target.js';
import { F32 } from '../../../_utils/ir_fixture.js';


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

function assertNoUseBeforeDef(func) {
  const defined = new Set(func.entryBlock.arguments);
  for (const op of func.entryBlock.ops()) {
    for (let i = 0; i < op.numOperands; i++) {
      const operand = op.getOperand(i);
      if (operand.isBlockArgument()) continue;
      expect(defined.has(operand)).toBe(true);
    }
    for (let i = 0; i < op.numResults; i++) defined.add(op.getResult(i));
  }
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
    return compileGraph(func, target, { fusion: { strategy: 'dominator' } });
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

  it('sum(softmax) with dominator equals 1 per row (no cyclic-fusion Inf)', () => {
    const inT = new TensorType([3, 4], F32);
    const outT = new TensorType([3], F32);
    const func = buildFunction('sum_softmax', [inT], [outT], (b, args) => {
      const zero = b.scalarConstant(0.0, F32);
      const sm = b.softmax(args[0], 1);
      const s = b.reduce(sm.getResult(0), zero.getResult(0), [1], 'sum');
      b.returnOp([s.getResult(0)]);
    });

    const result = compileWithDominator(func, CPUTarget());
    const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    const out = new Float32Array(3);
    result.run('sum_softmax', input, out);

    for (let i = 0; i < 3; i++) {
      expect(Number.isFinite(out[i])).toBe(true);
      expect(Math.abs(out[i] - 1)).toBeLessThan(1e-4);
    }
  });

  it('softmax(softmax) with dominator stays finite and matches a reference', () => {
    const t = new TensorType([3, 4], F32);
    const func = buildFunction('sm_sm', [t], [t], (b, args) => {
      b.returnOp([b.softmax(b.softmax(args[0], 1).getResult(0), 1).getResult(0)]);
    });
    const dom = compileWithDominator(func, CPUTarget());
    const base = compileGraph(func, CPUTarget(), {});
    const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    const outDom = new Float32Array(12);
    const outBase = new Float32Array(12);
    dom.run('sm_sm', input, outDom);
    base.run('sm_sm', input, outBase);

    for (let i = 0; i < 12; i++) {
      expect(Number.isFinite(outDom[i])).toBe(true);
      expect(Math.abs(outDom[i] - outBase[i])).toBeLessThan(1e-4);
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

describe('DominatorFusionPass — insertion point dominates group inputs', () => {
  it('does not insert fused op before a late-defined external input', () => {
    const t = new TensorType([8, 8], F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const early = b.exp(args[0]);
      const external = b.matmul(args[0], args[1]);
      const combined = b.add(early.getResult(0), external.getResult(0));
      b.returnOp([b.neg(combined.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    assertNoUseBeforeDef(func);
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
