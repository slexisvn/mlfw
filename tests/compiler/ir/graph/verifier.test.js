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

  it('accepts a value used before its definition in block order (dataflow DAG, no topological requirement)', () => {
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

  it('flags a value that escapes the region it is defined in', () => {
    const t = f32([4]);
    const pred = new TensorType([], ScalarType.BOOL);
    const func = buildFunction('escape', [pred, t], [t], (b, args) => {
      const branch = b.ifOp(
        args[0],
        [t],
        (tb) => { tb.yieldOp([tb.neg(args[1]).getResult(0)]); },
        (eb) => { eb.yieldOp([eb.exp(args[1]).getResult(0)]); }
      );
      b.returnOp([branch.getResult(0)]);
    });
    const negOp = [...func.opsRecursive()].find(o => o.opName === 'neg');
    const innerValue = negOp.getResult(0);
    func.getReturnOp().replaceOperand(0, innerValue);

    const errors = verifyFunction(func);
    expect(errors.some(e => /used before definition/.test(e.message))).toBe(true);
  });

  it('flags a value dependency cycle', () => {
    const t = f32([4]);
    const func = buildFunction('cycle', [t], [t], (b, args) => {
      const a = b.add(args[0], args[0]);
      const c = b.mul(a.getResult(0), args[0]);
      b.returnOp([c.getResult(0)]);
    });
    const addOp = func.findOp(o => o.opName === 'add');
    const mulOp = func.findOp(o => o.opName === 'mul');
    addOp.replaceOperand(0, mulOp.getResult(0));

    const errors = verifyFunction(func);
    expect(errors.some(e => /dependency cycle/.test(e.message))).toBe(true);
  });

  it('still accepts a value defined in an enclosing scope from a nested region', () => {
    const t = f32([4]);
    const func = buildFunction('nested', [t], [t], (b, args) => {
      const outer = b.add(args[0], args[0]);
      const pred = b.scalarConstant(1);
      const branch = b.ifOp(
        pred.getResult(0),
        [t],
        (tb) => { tb.yieldOp([outer.getResult(0)]); },
        (eb) => { eb.yieldOp([outer.getResult(0)]); }
      );
      b.returnOp([branch.getResult(0)]);
    });
    const errors = verifyFunction(func);
    expect(errors.some(e => /used before definition/.test(e.message))).toBe(false);
  });
});
