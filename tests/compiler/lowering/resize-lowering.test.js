import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { BufferLoadNode, CastNode } from '../../../src/compiler/ir/tensor/nodes.js';

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

function lower(inT, outT, method) {
  const func = buildFunction('f', [inT], [outT], (b, args) => {
    b.returnOp([b.resize(args[0], [outT.shape[2], outT.shape[3]], method).getResult(0)]);
  });
  return lowerGraphToPrimFunc(func);
}

describe('resize lowering index integrality', () => {
  it('bilinear resize wraps all spatial buffer indices in integer CastNodes', () => {
    const inT = new TensorType([1, 3, 4, 4], ScalarType.F32);
    const outT = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const pf = lower(inT, outT, 'bilinear');

    const loads = collectNodes(pf.body, n => n instanceof BufferLoadNode);
    const inputLoads = loads.filter(l => l.indices.length === 4);
    expect(inputLoads.length).toBeGreaterThan(0);

    for (const load of inputLoads) {
      const spatial = [load.indices[2], load.indices[3]];
      for (const idx of spatial) {
        expect(idx).toBeInstanceOf(CastNode);
        expect(idx.toDtype).toBe('i32');
      }
    }
  });

  it('nearest resize keeps integer-cast spatial indices', () => {
    const inT = new TensorType([1, 3, 4, 4], ScalarType.F32);
    const outT = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const pf = lower(inT, outT, 'nearest');

    const loads = collectNodes(pf.body, n => n instanceof BufferLoadNode);
    const inputLoads = loads.filter(l => l.indices.length === 4);
    for (const load of inputLoads) {
      expect(load.indices[2]).toBeInstanceOf(CastNode);
      expect(load.indices[3]).toBeInstanceOf(CastNode);
    }
  });
});
