import { describe, it, expect } from 'vitest';
import {
  tensor, trace,
  Linear,
} from '../../src/index.js';

function getFunc(graph) {
  return graph.functions().next().value;
}

function collectOps(func) {
  const names = [];
  for (const op of func.entryBlock.ops()) names.push(op.opName);
  return names;
}

function getOp(func, name) {
  for (const op of func.entryBlock.ops()) {
    if (op.opName === name) return op;
  }
  return null;
}

describe('dispatch scalar arg spec parsing', () => {
  it('sum with explicit dim traces reduce on that axis only', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const func = getFunc(trace((t) => t.sum([1]), [x]));
    const reduce = getOp(func, 'reduce');
    expect(reduce).not.toBeNull();
    expect(reduce.getAttr('dimensions')).toEqual([1]);
  });

  it('sum without dim reduces all axes', () => {
    const x = tensor([[1, 2], [3, 4]]);
    const func = getFunc(trace((t) => t.sum(), [x]));
    const reduce = getOp(func, 'reduce');
    expect(reduce.getAttr('dimensions')).toEqual([0, 1]);
  });

  it('mean without dim reduces all axes', () => {
    const x = tensor([[1, 2], [3, 4]]);
    const func = getFunc(trace((t) => t.mean(), [x]));
    const reduce = getOp(func, 'reduce');
    expect(reduce.getAttr('dimensions')).toEqual([0, 1]);
    expect(reduce.getAttr('reduce_type')).toBe('mean');
  });

  it('sum with null dim and keepdim does not misalign keepdim into dim slot', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]]);
    const func = getFunc(trace((t) => t.sum(null, true), [x]));
    const reduce = getOp(func, 'reduce');
    expect(reduce.getAttr('dimensions')).toEqual([0, 1]);
  });
});

describe('recordOp fallback chain', () => {
  it('add goes through shared IR mapping fallback', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const func = getFunc(trace((x, y) => x.add(y), [a, b]));
    expect(collectOps(func)).toContain('add');
  });

  it('sub goes through builder method fallback', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const func = getFunc(trace((x, y) => x.sub(y), [a, b]));
    expect(collectOps(func)).toContain('sub');
  });

  it('mul goes through builder method fallback', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const func = getFunc(trace((x, y) => x.mul(y), [a, b]));
    expect(collectOps(func)).toContain('mul');
  });

  it('neg goes through builder method fallback', () => {
    const a = tensor([1, 2]);
    const func = getFunc(trace((x) => x.neg(), [a]));
    expect(collectOps(func)).toContain('neg');
  });

  it('exp falls through to _inferAndBuild', () => {
    const a = tensor([1, 2]);
    const func = getFunc(trace((x) => x.exp(), [a]));
    expect(collectOps(func)).toContain('exp');
  });

  it('log falls through to _inferAndBuild', () => {
    const a = tensor([1, 2]);
    const func = getFunc(trace((x) => x.log(), [a]));
    expect(collectOps(func)).toContain('log');
  });
});

describe('non-symbolic tensor capture during tracing', () => {
  it('real tensor arg is captured as extra input via captureConstant', () => {
    const x = tensor([1, 2, 3, 4]);
    const weight = tensor([2, 2, 2, 2]);
    const func = getFunc(trace((inp) => inp.mul(weight), [x]));
    expect(func.inputTypes.length).toBe(2);
    expect(collectOps(func)).toContain('mul');
  });

  it('same tensor captured twice still produces one extra input', () => {
    const x = tensor([1, 2]);
    const w = tensor([3, 4]);
    const func = getFunc(trace((inp) => inp.mul(w).add(w), [x]));
    expect(func.inputTypes.length).toBe(2);
  });
});

describe('compare ops trace correctly', () => {
  it('eq produces compare op with eq direction', () => {
    const a = tensor([1, 2]);
    const b = tensor([1, 3]);
    const func = getFunc(trace((x, y) => x.eq(y), [a, b]));
    const cmp = getOp(func, 'compare');
    expect(cmp).not.toBeNull();
    expect(cmp.getAttr('direction')).toBe('eq');
  });

  it('lt produces compare op with lt direction', () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const func = getFunc(trace((x, y) => x.lt(y), [a, b]));
    const cmp = getOp(func, 'compare');
    expect(cmp.getAttr('direction')).toBe('lt');
  });
});

describe('multi-output tracing with markOutputs', () => {
  it('function returning array produces multiple output types', () => {
    const x = tensor([1, 2, 3]);
    const func = getFunc(trace((t) => [t.neg(), t.exp()], [x]));
    expect(func.outputTypes.length).toBe(2);
    expect(collectOps(func)).toContain('neg');
    expect(collectOps(func)).toContain('exp');
  });
});
