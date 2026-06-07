import { describe, it, expect } from 'vitest';
import { flattenIndex, computeDynamicStride, resolveShapeParam, computeNumelExpr } from '../../../../src/compiler/ir/lir/flatten.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { VariableNode, IntImmNode, MathOpNode } from '../../../../src/compiler/ir/tensor/nodes.js';

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function evalExpr(node) {
  if (node.type === 'IntImmNode') return node.value;
  if (node.type === 'MathOpNode') {
    const a = evalExpr(node.a);
    const b = evalExpr(node.b);
    switch (node.op) {
      case '+': return a + b;
      case '*': return a * b;
      case '-': return a - b;
    }
  }
  return NaN;
}

function exprToString(node) {
  if (!node) return '?';
  if (node.type === 'IntImmNode') return String(node.value);
  if (node.type === 'VariableNode') return node.name;
  if (node.type === 'MathOpNode') {
    if (!node.b) return `(${node.op}${exprToString(node.a)})`;
    return `(${exprToString(node.a)} ${node.op} ${exprToString(node.b)})`;
  }
  return `[${node.type}]`;
}

describe('flattenIndex', () => {
  it('returns IntImmNode(0) for empty indices', () => {
    const b = buf('s', []);
    const result = flattenIndex(b, [], null);
    expect(result.type).toBe('IntImmNode');
    expect(result.value).toBe(0);
  });

  it('returns the index directly for 1D', () => {
    const b = buf('v', [8]);
    const i = idx('i');
    const result = flattenIndex(b, [i], null);
    expect(result).toBe(i);
  });

  it('computes flat index for 2D: i*4 + j', () => {
    const b = buf('m', [3, 4]);
    const result = flattenIndex(b, [idx('i'), idx('j')], null);
    expect(result.type).toBe('MathOpNode');
    expect(result.op).toBe('+');
    const str = exprToString(result);
    expect(str).toContain('i');
    expect(str).toContain('j');
    expect(str).toContain('4');
  });

  it('computes flat index for 3D: i*12 + j*4 + k', () => {
    const b = buf('t', [2, 3, 4]);
    const result = flattenIndex(b, [idx('i'), idx('j'), idx('k')], null);
    const str = exprToString(result);
    expect(str).toContain('12');
    expect(str).toContain('4');
  });

  it('skips zero indices', () => {
    const b = buf('m', [3, 4]);
    const result = flattenIndex(b, [new IntImmNode(0), idx('j')], null);
    expect(result.type).toBe('VariableNode');
    expect(result.name).toBe('j');
  });

  it('omits stride=1 multiplication', () => {
    const b = buf('m', [3, 4]);
    const result = flattenIndex(b, [idx('i'), idx('j')], null);
    const str = exprToString(result);
    expect(str).not.toMatch(/j \* 1\b/);
  });

  it('handles dynamic stride with shapeParamMap', () => {
    const shapeParamMap = new Map([['d:1', { name: 'n' }]]);
    const b = buf('d', [3, -1]);
    b.strides = [-1, 1];
    const result = flattenIndex(b, [idx('i'), idx('j')], shapeParamMap);
    const str = exprToString(result);
    expect(str).toContain('n');
  });

  it('evaluates constant 2D correctly: idx=(2,1) shape=(3,4) -> 9', () => {
    const b = buf('m', [3, 4]);
    const result = flattenIndex(b, [new IntImmNode(2), new IntImmNode(1)], null);
    expect(evalExpr(result)).toBe(9);
  });

  it('evaluates constant 3D correctly: idx=(1,2,3) shape=(2,3,4) -> 23', () => {
    const b = buf('t', [2, 3, 4]);
    const result = flattenIndex(b, [new IntImmNode(1), new IntImmNode(2), new IntImmNode(3)], null);
    expect(evalExpr(result)).toBe(23);
  });
});

describe('computeDynamicStride', () => {
  it('returns 1 for last dimension', () => {
    const b = buf('v', [8]);
    const result = computeDynamicStride(b, 0, null);
    expect(result.type).toBe('IntImmNode');
    expect(result.value).toBe(1);
  });

  it('returns trailing dim for second-to-last', () => {
    const b = buf('m', [3, 4]);
    const result = computeDynamicStride(b, 0, null);
    expect(result.type).toBe('IntImmNode');
    expect(result.value).toBe(4);
  });

  it('returns product for 3D first dim', () => {
    const b = buf('t', [2, 3, 4]);
    const result = computeDynamicStride(b, 0, null);
    expect(evalExpr(result)).toBe(12);
  });

  it('resolves dynamic shape via shapeParamMap', () => {
    const shapeParamMap = new Map([['d:1', { name: 'n' }]]);
    const b = buf('d', [3, -1]);
    const result = computeDynamicStride(b, 0, shapeParamMap);
    expect(result.type).toBe('VariableNode');
    expect(result.name).toBe('n');
  });
});

describe('resolveShapeParam', () => {
  it('returns VariableNode when param exists', () => {
    const shapeParamMap = new Map([['x:0', { name: 'batch' }]]);
    const b = buf('x', [10]);
    const result = resolveShapeParam(b, 0, shapeParamMap);
    expect(result.type).toBe('VariableNode');
    expect(result.name).toBe('batch');
  });

  it('returns IntImmNode(1) when no param', () => {
    const b = buf('x', [10]);
    const result = resolveShapeParam(b, 0, null);
    expect(result.type).toBe('IntImmNode');
    expect(result.value).toBe(1);
  });

  it('returns IntImmNode(1) when key not in map', () => {
    const shapeParamMap = new Map([['y:0', { name: 'n' }]]);
    const b = buf('x', [10]);
    const result = resolveShapeParam(b, 0, shapeParamMap);
    expect(result.type).toBe('IntImmNode');
    expect(result.value).toBe(1);
  });
});

describe('computeNumelExpr', () => {
  it('returns 1 for scalar', () => {
    const b = buf('s', []);
    const result = computeNumelExpr(b, null);
    expect(result.type).toBe('IntImmNode');
    expect(result.value).toBe(1);
  });

  it('returns correct product for static shape', () => {
    const b = buf('m', [3, 4]);
    const result = computeNumelExpr(b, null);
    expect(evalExpr(result)).toBe(12);
  });

  it('includes dynamic shape params', () => {
    const shapeParamMap = new Map([['d:0', { name: 'batch' }]]);
    const b = buf('d', [-1, 4]);
    const result = computeNumelExpr(b, shapeParamMap);
    const str = exprToString(result);
    expect(str).toContain('batch');
    expect(str).toContain('4');
  });
});
