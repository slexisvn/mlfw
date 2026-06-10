import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { verifyFunction } from '../../../../src/compiler/ir/graph/verifier.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';

function f32(shape) {
  return new TensorType(shape, ScalarType.F32);
}

describe('verifyFunction operand definition', () => {
  it('accepts valid IR', () => {
    const t = f32([4]);
    const func = buildFunction('valid', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([sum.getResult(0)]);
    });
    expect(verifyFunction(func).length).toBe(0);
  });

  it('flags an operand that is not defined anywhere in the function', () => {
    const t = f32([4]);
    const func = buildFunction('dangling', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      b.returnOp([sum.getResult(0)]);
    });
    const otherFunc = buildFunction('other', [t], [t], (b, args) => {
      b.returnOp([args[0]]);
    });
    const foreignValue = otherFunc.args[0];
    func.findOp(o => o.opName === 'add').replaceOperand(0, foreignValue);

    const errors = verifyFunction(func);
    expect(errors.some(e => /used before definition/.test(e.message))).toBe(true);
  });

  it('accepts a value defined later in block order (no topological requirement)', () => {
    const t = f32([4]);
    const func = buildFunction('reorder', [t, t], [t], (b, args) => {
      const a = b.add(args[0], args[1]);
      const c = b.mul(a.getResult(0), args[0]);
      b.returnOp([c.getResult(0)]);
    });
    const block = func.entryBlock;
    const addOp = func.findOp(o => o.opName === 'add');
    const mulOp = func.findOp(o => o.opName === 'mul');
    block.removeOp(addOp);
    block.insertAfter(addOp, mulOp);

    const errors = verifyFunction(func);
    expect(errors.some(e => /used before definition/.test(e.message))).toBe(false);
  });
});
