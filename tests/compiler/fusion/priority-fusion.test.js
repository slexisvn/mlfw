import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { PriorityFusionPass } from '../../../src/compiler/passes/fusion/priority_fusion.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../src/backend/target.js';

const F32 = ScalarType.F32;

function run(func, config = {}) {
  return new PriorityFusionPass(config).run(func);
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

describe('PriorityFusionPass — grouping', () => {
  it('fuses an elementwise chain into one group', () => {
    const t = new TensorType([64, 64], F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const a = b.neg(sum.getResult(0));
      b.returnOp([b.exp(a.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);
    expect(findOps(func, 'fusion').length).toBe(1);
    assertNoUseBeforeDef(func);
  });

  it('keeps at most one reduction per group', () => {
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
    for (const f of findOps(func, 'fusion')) {
      expect(countReductionsInGroup(f)).toBeLessThanOrEqual(1);
    }
    assertNoUseBeforeDef(func);
  });

  it('produces valid IR on a diamond (multi-consumer producer)', () => {
    const t = new TensorType([16, 16], F32);
    const func = buildFunction('d', [t, t], [t], (b, args) => {
      const base = b.add(args[0], args[1]);
      const left = b.exp(base.getResult(0));
      const right = b.neg(base.getResult(0));
      b.returnOp([b.mul(left.getResult(0), right.getResult(0)).getResult(0)]);
    });

    run(func);
    assertNoUseBeforeDef(func);
  });

  it('reports UNCHANGED when nothing is fusible', () => {
    const t = new TensorType([8, 8], F32);
    const func = buildFunction('id', [t], [t], (b, args) => {
      b.returnOp([b.matmul(args[0], args[0]).getResult(0)]);
    });
    expect(run(func)).toBe(PassResult.UNCHANGED);
  });
});

describe('PriorityFusionPass — compiled correctness', () => {
  function compileWithPriority(func, target) {
    return compileGraph(func, target, { fusion: { strategy: 'priority' } });
  }

  it('matches the default pipeline on an elementwise chain (CPU)', () => {
    const t = new TensorType([3, 4], F32);
    const func = buildFunction('chain', [t, t], [t], (b, args) => {
      const s = b.add(args[0], args[1]);
      const n = b.neg(s.getResult(0));
      b.returnOp([b.exp(n.getResult(0)).getResult(0)]);
    });

    const prio = compileWithPriority(func, CPUTarget());
    const base = compileGraph(func, CPUTarget(), {});
    const a = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    const bb = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    const outP = new Float32Array(12);
    const outB = new Float32Array(12);
    prio.run('chain', a, bb, outP);
    base.run('chain', a, bb, outB);
    for (let i = 0; i < 12; i++) {
      expect(Number.isFinite(outP[i])).toBe(true);
      expect(Math.abs(outP[i] - outB[i])).toBeLessThan(1e-5);
    }
  });

  it('softmax with priority fusion matches reference and stays finite (CPU)', () => {
    const t = new TensorType([3, 4], F32);
    const func = buildFunction('sm', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0], 1).getResult(0)]);
    });

    const prio = compileWithPriority(func, CPUTarget());
    const base = compileGraph(func, CPUTarget(), {});
    const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    const outP = new Float32Array(12);
    const outB = new Float32Array(12);
    prio.run('sm', input, outP);
    base.run('sm', input, outB);
    for (let i = 0; i < 12; i++) {
      expect(Number.isFinite(outP[i])).toBe(true);
      expect(Math.abs(outP[i] - outB[i])).toBeLessThan(1e-4);
    }
  });

  it('reduce+elementwise chain with priority fusion matches reference (WASM)', () => {
    const t = new TensorType([8, 16], F32);
    const outT = new TensorType([8], F32);
    const func = buildFunction('rchain', [t], [outT], (b, args) => {
      const zero = b.scalarConstant(0.0, F32);
      const s = b.add(args[0], args[0]);
      b.returnOp([b.reduce(s.getResult(0), zero.getResult(0), [1], 'sum').getResult(0)]);
    });

    const prio = compileWithPriority(func, WasmTarget());
    const base = compileGraph(func, WasmTarget(), {});
    const input = new Float32Array(128);
    for (let i = 0; i < 128; i++) input[i] = (i % 7) * 0.1 - 0.3;
    const outP = new Float32Array(8);
    const outB = new Float32Array(8);
    prio.run('rchain', input, outP);
    base.run('rchain', input, outB);
    for (let i = 0; i < 8; i++) {
      expect(Math.abs(outP[i] - outB[i])).toBeLessThan(1e-4);
    }
  });
});
