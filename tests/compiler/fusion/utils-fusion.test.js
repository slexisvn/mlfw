import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import {
  getYieldOp, countInnerOps, countReductions, allInnerOpsFusable, remapOperands
} from '../../../src/compiler/passes/fusion/fusion_utils.js';
import { FusionPass } from '../../../src/compiler/passes/fusion/fusion_pass.js';
import { Operation } from '../../../src/compiler/ir/graph/operation.js';
import { Block, Region } from '../../../src/compiler/ir/graph/block.js';

function findOps(func, opName) {
  const result = [];
  for (const op of func.ops()) {
    if (op.opName === opName) result.push(op);
  }
  return result;
}

function makeFused(func) {
  new FusionPass().run(func);
  return findOps(func, 'fusion')[0];
}

describe('getYieldOp', () => {
  it('returns the yield op from a fusion body', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const fusionOp = makeFused(func);
    const block = fusionOp.regions[0].entryBlock;
    const yld = getYieldOp(block);
    expect(yld).toBeDefined();
    expect(yld.opName).toBe('yield');
  });
});

describe('countInnerOps', () => {
  it('counts non-yield ops inside fusion', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const fusionOp = makeFused(func);
    expect(countInnerOps(fusionOp)).toBe(2);
  });

  it('3-op chain counts 3', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const n = b.neg(sum.getResult(0));
      b.returnOp([b.exp(n.getResult(0)).getResult(0)]);
    });
    const fusionOp = makeFused(func);
    expect(countInnerOps(fusionOp)).toBe(3);
  });
});

describe('countReductions', () => {
  it('counts reduction ops inside fusion', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const bodyRegion = new Region();
    const bodyBlock = new Block([t, t]);
    bodyRegion.addBlock(bodyBlock);
    const addOp = new Operation('add', [bodyBlock.arguments[0], bodyBlock.arguments[1]], [t]);
    bodyBlock.pushOp(addOp);
    const initType = new TensorType([], ScalarType.F32);
    const initBlock = new Block([initType, initType]);
    const initRegion = new Region();
    initRegion.addBlock(initBlock);
    const reduceOp = new Operation('reduce', [addOp.getResult(0)], [outT], { dimensions: [1], kind: 'sum' }, [initRegion]);
    bodyBlock.pushOp(reduceOp);
    const yieldOp = new Operation('yield', [reduceOp.getResult(0)], []);
    bodyBlock.pushOp(yieldOp);
    const fusionOp = new Operation('fusion', [], [outT], {}, [bodyRegion]);
    expect(countReductions(fusionOp)).toBe(1);
  });

  it('returns 0 for pure elementwise fusion', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const fusionOp = makeFused(func);
    expect(countReductions(fusionOp)).toBe(0);
  });
});

describe('allInnerOpsFusable', () => {
  it('returns true for all elementwise ops', () => {
    const t = new TensorType([8], ScalarType.F32);
    const func = buildFunction('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([b.neg(sum.getResult(0)).getResult(0)]);
    });
    const fusionOp = makeFused(func);
    expect(allInnerOpsFusable(fusionOp)).toBe(true);
  });
});
