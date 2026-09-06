import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType, Layout } from '../../../../src/compiler/ir/graph/types.js';
import { CanonicalizePass } from '../../../../src/compiler/passes/canonicalize/canonicalize.js';
import { DCEPass } from '../../../../src/compiler/passes/simplify/dce.js';

function run(func) {
  const result = new CanonicalizePass().run(func);
  new DCEPass().run(func);
  return result;
}

function transformsOf(func) {
  const out = [];
  for (const op of func.ops()) {
    if (op.opName === 'layout_transform') out.push(op);
  }
  return out;
}

function chain(b, value, steps) {
  let current = value;
  for (const [src, dst] of steps) {
    current = b._inferAndBuild('layout_transform', [current], { src_layout: src, dst_layout: dst }).getResult(0);
  }
  return current;
}

describe('layout_transform canonicalization', () => {
  it('drops a transform that asks for the layout the value already has', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([chain(b, args[0], [[[0, 1], [0, 1]]])]);
    });

    run(func);

    expect(transformsOf(func).length).toBe(0);
    expect(func.getReturnOp().getOperand(0)).toBe(func.args[0]);
  });

  it('collapses a chain of two transforms into the outermost layout', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([chain(b, args[0], [[[0, 1], [1, 0]], [[1, 0], [0, 1]]])]);
    });

    run(func);

    expect(transformsOf(func).length).toBe(0);
    expect(func.getReturnOp().getOperand(0)).toBe(func.args[0]);
  });

  it('keeps the source and destination of the chain it collapses', () => {
    const t = new TensorType([2, 4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([chain(b, args[0], [[[0, 1, 2], [2, 0, 1]], [[2, 0, 1], [1, 2, 0]]])]);
    });

    run(func);

    const transforms = transformsOf(func);
    expect(transforms.length).toBe(1);
    expect(transforms[0].getAttr('src_layout')).toEqual([0, 1, 2]);
    expect(transforms[0].getAttr('dst_layout')).toEqual([1, 2, 0]);
    expect(transforms[0].getOperand(0)).toBe(func.args[0]);
  });

  it('gives the collapsed transform the layout its result type claims', () => {
    const t = new TensorType([2, 4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([chain(b, args[0], [[[0, 1, 2], [2, 0, 1]], [[2, 0, 1], [1, 2, 0]]])]);
    });

    run(func);

    const result = transformsOf(func)[0].getResult(0);
    expect(result.type.layout.equals(new Layout([1, 2, 0]))).toBe(true);
  });

  it('leaves a chain alone when the middle layouts disagree', () => {
    const t = new TensorType([2, 4, 8], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      b.returnOp([chain(b, args[0], [[[0, 1, 2], [2, 0, 1]], [[0, 1, 2], [1, 2, 0]]])]);
    });

    run(func);

    expect(transformsOf(func).length).toBe(2);
  });
});
