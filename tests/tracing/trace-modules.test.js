import { describe, it, expect } from 'vitest';
import {
  tensor, trace, fromBuffer,
  Linear, MaxPool2d,
  ReLU, Sigmoid, GELU,
  Sequential,
} from '../../src/index.js';
import { ScalarType } from '../../src/compiler/ir/graph/types.js';

function t4d(data, shape) {
  return fromBuffer(new Float32Array(data), shape, ScalarType.F32);
}

function traceModule(mod, ...inputs) {
  return trace((...args) => mod.forward(...args), inputs, { name: mod.constructor.name });
}

function getFunc(graph) {
  return graph.functions().next().value;
}

function collectOps(func) {
  const names = [];
  for (const op of func.entryBlock.ops()) names.push(op.opName);
  return names;
}

describe('pool2d builder method map produces correct graph ops', () => {
  it('MaxPool2d traces to pool2d op with correct attributes', () => {
    const pool = new MaxPool2d(2);
    const x = t4d(new Array(16).fill(0), [1, 1, 4, 4]);
    const func = getFunc(traceModule(pool, x));

    let poolOp = null;
    for (const op of func.entryBlock.ops()) {
      if (op.opName === 'pool2d') poolOp = op;
    }
    expect(poolOp).not.toBeNull();
    expect(poolOp.getAttr('pool_type')).toBe('max');
    expect(poolOp.getAttr('kernel_size')).toEqual([2, 2]);
    expect(poolOp.getAttr('strides')).toEqual([2, 2]);
  });

  it('MaxPool2d with custom kernel and stride traces correct attributes', () => {
    const pool = new MaxPool2d(3, 2);
    const x = t4d(new Array(36).fill(0), [1, 1, 6, 6]);
    const func = getFunc(traceModule(pool, x));

    let poolOp = null;
    for (const op of func.entryBlock.ops()) {
      if (op.opName === 'pool2d') poolOp = op;
    }
    expect(poolOp.getAttr('kernel_size')).toEqual([3, 3]);
    expect(poolOp.getAttr('strides')).toEqual([2, 2]);
  });
});

describe('scalar constant inlining vs parameter capture', () => {
  it('non-scalar weight is captured as extra graph input', () => {
    const layer = new Linear(4, 2);
    const x = tensor([[1, 1, 1, 1]]);
    const func = getFunc(traceModule(layer, x));
    expect(func.inputTypes.length).toBe(3);
  });

  it('reduce init scalar is inlined as constant op, not captured', () => {
    const x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
    const func = getFunc(trace((t) => t.sum([1]), [x], { name: 'sum_trace' }));

    expect(collectOps(func).includes('constant')).toBe(true);
    expect(func.inputTypes.length).toBe(1);
  });
});

describe('Sequential tracing flattens multi-layer model into one graph', () => {
  it('Linear+ReLU+Linear produces dot and relu ops', () => {
    const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 2));
    const x = tensor([[1, 1, 1, 1]]);
    const ops = new Set(collectOps(getFunc(traceModule(model, x))));
    expect(ops.has('dot')).toBe(true);
    expect(ops.has('relu') || ops.has('maximum')).toBe(true);
  });
});

describe('activation op names in traced graph', () => {
  it('sigmoid traces to sigmoid op', () => {
    const x = tensor([[0, 0, 0, 0]]);
    expect(collectOps(getFunc(traceModule(new Sigmoid(), x))).includes('sigmoid')).toBe(true);
  });

  it('gelu traces to gelu op', () => {
    const x = tensor([[0, 0, 0, 0]]);
    expect(collectOps(getFunc(traceModule(new GELU(), x))).includes('gelu')).toBe(true);
  });
});

describe('parameter deduplication across repeated forward calls', () => {
  it('same weight tensor reused twice maps to one captured param', () => {
    const layer = new Linear(4, 4);
    const x = tensor([[1, 1, 1, 1]]);
    const func = getFunc(trace((inp) => layer.forward(layer.forward(inp)), [x], { name: 'shared' }));
    expect(func.inputTypes.length).toBe(3);
  });
});
