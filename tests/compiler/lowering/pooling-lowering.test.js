import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { BufferLoadNode, IntImmNode } from '../../../src/compiler/ir/tensor/nodes.js';

const SKIP_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);

function collectNodes(node, predicate) {
  const result = [];
  const visited = new Set();
  const stack = [node];
  while (stack.length > 0) {
    const n = stack.pop();
    if (!n || typeof n !== 'object' || visited.has(n)) continue;
    visited.add(n);
    if (predicate(n)) result.push(n);
    for (const key of Object.keys(n)) {
      if (SKIP_KEYS.has(key)) continue;
      const val = n[key];
      if (Array.isArray(val)) for (const item of val) stack.push(item);
      else if (val && typeof val === 'object') stack.push(val);
    }
  }
  return result;
}

function loadFor(pf, bufferName) {
  return collectNodes(pf.body, n => n instanceof BufferLoadNode && n.buffer && n.buffer.name === bufferName);
}

function inputLoad(pf, inBufName) {
  const loads = loadFor(pf, inBufName);
  return loads.find(l => l.indices.length === 4);
}

describe('pool2d lowering layout awareness', () => {
  it('NCHW max-pool indexes channels at axis 1 and spatial at axes 2,3', () => {
    const inT = new TensorType([1, 3, 4, 4], ScalarType.F32);
    const outT = new TensorType([1, 3, 2, 2], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      b.returnOp([b.pool2d(args[0], 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);
    const inBuf = pf.bufferMap.get(pf.params[0]);
    const load = inputLoad(pf, inBuf.name);
    expect(load).toBeDefined();
    expect(load.buffer.shape).toEqual([1, 3, 4, 4]);
  });

  it('NHWC avg-pool reads spatial dims from layout axes 1,2 (not hardcoded 2,3)', () => {
    const inT = new TensorType([1, 4, 4, 3], ScalarType.F32);
    const outT = new TensorType([1, 2, 2, 3], ScalarType.F32);
    const func = buildFunction('f', [inT], [outT], (b, args) => {
      const op = b._buildOp('pool2d', [args[0]], [outT], {
        pool_type: 'avg',
        kernel_size: [2, 2],
        strides: [2, 2],
        padding: [[0, 0], [0, 0]],
        ceil_mode: false,
        count_include_pad: false,
        layout: 'NHWC',
      });
      b.returnOp([op.getResult(0)]);
    });
    const pf = lowerGraphToPrimFunc(func);
    const inBuf = pf.bufferMap.get(pf.params[0]);
    const load = inputLoad(pf, inBuf.name);
    expect(load).toBeDefined();

    expect(load.indices.length).toBe(4);
    const chIdx = load.indices[3];
    const isPlainVar = chIdx && chIdx.type !== 'MathOpNode';
    expect(isPlainVar).toBe(true);

    const stores = collectNodes(pf.body, n => n && n.type === 'BufferStoreNode');
    expect(stores.length).toBeGreaterThan(0);
  });
});
