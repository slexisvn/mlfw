import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { FusionPass } from '../../../src/compiler/passes/fusion/fusion_pass.js';
import { PassResult } from '../../../src/compiler/passes/pass.js';

function run(func, config = {}) {
  return new FusionPass(config).run(func);
}

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

describe('FusionPass end-to-end', () => {
  it('fuses add -> neg into a single fusion op', () => {
    const t = new TensorType([64, 64], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.CHANGED);

    const fusions = findOps(func, 'fusion');
    expect(fusions.length).toBe(1);
    expect(fusions[0].numOperands).toBe(2);
    expect(fusions[0].numResults).toBe(1);

    expect(findOps(func, 'add').length).toBe(0);
    expect(findOps(func, 'neg').length).toBe(0);
  });

  it('fusion body contains cloned ops and yield', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    const body = fusions[0].regions[0].entryBlock;
    const innerOps = [];
    for (const op of body.ops()) innerOps.push(op);

    expect(innerOps.length).toBe(3);
    expect(innerOps[0].opName).toBe('add');
    expect(innerOps[1].opName).toBe('neg');
    expect(innerOps[2].opName).toBe('yield');
  });

  it('return op uses fusion result', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    run(func);

    const ret = func.getReturnOp();
    expect(ret.getOperand(0).definingOp.opName).toBe('fusion');
  });

  it('returns UNCHANGED when nothing to fuse', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    expect(run(func)).toBe(PassResult.UNCHANGED);
  });

  it('three-op chain fuses into one fusion', () => {
    const t = new TensorType([32, 32], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n = b.neg(sum.getResult(0));
      b.returnOp([b.exp(n.getResult(0)).getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    expect(fusions.length).toBe(1);

    const body = fusions[0].regions[0].entryBlock;
    let opCount = 0;
    for (const op of body.ops()) {
      if (op.opName !== 'yield') opCount++;
    }
    expect(opCount).toBe(3);
  });

  it('preserves ops that are not part of any fusion group', () => {
    const t = new TensorType([64], ScalarType.F32);
    const t2 = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t, t2], [t, t2], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n = b.neg(sum.getResult(0));
      const standalone = b.neg(args[2]);
      b.returnOp([n.getResult(0), standalone.getResult(0)]);
    });

    run(func);

    const standaloneNegs = findOps(func, 'neg');
    expect(standaloneNegs.length).toBe(1);
  });

  it('fusion op operands match group input values', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    expect(fusions[0].getOperand(0).isBlockArgument()).toBe(true);
    expect(fusions[0].getOperand(1).isBlockArgument()).toBe(true);
  });
});

describe('FusionPass with elementwise + reduction', () => {
  it('does not fuse elementwise into reduction — reductions stay standalone', () => {
    const t = new TensorType([32, 64], ScalarType.F32);
    const s = new TensorType([], ScalarType.F32);
    const outT = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [t, t, s], [outT], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.reduce(sum.getResult(0), args[2], [1], 'sum').getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    expect(fusions.length).toBe(0);
    expect(findOps(func, 'add').length).toBe(1);
    expect(findOps(func, 'reduce').length).toBe(1);
  });
});

describe('regression — fusion insertion respects data dependency ordering', () => {
  it('fusion op is inserted after all its input producers', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('f', [t, t, t], [t], (b, args) => {
      const a = b.neg(args[0]);
      const c = b.neg(args[2]);
      b.returnOp([b.add(a.getResult(0), c.getResult(0)).getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    if (fusions.length === 0) return;

    const fusionOp = fusions[0];
    for (let i = 0; i < fusionOp.numOperands; i++) {
      const producer = fusionOp.getOperand(i).definingOp;
      if (!producer || producer === fusionOp) continue;
      let foundProducer = false;
      let foundFusion = false;
      let cur = fusionOp.parentBlock.firstOp;
      while (cur) {
        if (cur === producer) foundProducer = true;
        if (cur === fusionOp) { foundFusion = true; break; }
        cur = cur._next;
      }
      expect(foundProducer).toBe(true);
      expect(foundFusion).toBe(true);
    }
  });

  it('fusion of ops with inputs from separate producers preserves ordering', () => {
    const v = new TensorType([32], ScalarType.F32);
    const func = buildFunction('f', [v, v], [v], (b, args) => {
      const scaled = b.mul(args[0], args[1]);
      const biased = b.add(scaled.getResult(0), args[1]);
      b.returnOp([biased.getResult(0)]);
    });

    run(func);

    const fusions = findOps(func, 'fusion');
    expect(fusions.length).toBe(1);

    const ret = func.getReturnOp();
    expect(ret.getOperand(0).definingOp.opName).toBe('fusion');
  });
});
