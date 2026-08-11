import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, IntImmNode, SeqNode, CastNode, IfThenElseNode, CompareNode
} from '../../../../src/compiler/ir/tensor/nodes.js';

function lower(name, inTypes, outTypes, bodyFn) {
  const func = buildFunction(name, inTypes, outTypes, bodyFn);
  return lowerGraphToPrimFunc(func);
}

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
      if (Array.isArray(val)) {
        for (const item of val) stack.push(item);
      } else if (val && typeof val === 'object') {
        stack.push(val);
      }
    }
  }
  return result;
}

function getLoopExtents(node) {
  const extents = [];
  let cur = node;
  while (cur instanceof ForNode) {
    extents.push(cur.extent.value);
    cur = cur.body;
  }
  return extents;
}

describe('transpose lowering', () => {
  it('permutation [1, 0] swaps index mapping between in and out buffers', () => {
    const inT = new TensorType([3, 5], ScalarType.F32);
    const outT = new TensorType([5, 3], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.transpose(args[0], [1, 0]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(1);
    const store = stores[0];
    expect(store.value).toBeInstanceOf(BufferLoadNode);

    expect(getLoopExtents(pf.body)).toEqual([5, 3]);
  });

  it('3D permutation [2, 0, 1] produces correct loop extents', () => {
    const inT = new TensorType([2, 3, 4], ScalarType.F32);
    const outT = new TensorType([4, 2, 3], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.transpose(args[0], [2, 0, 1]).getResult(0)]);
    });

    expect(getLoopExtents(pf.body)).toEqual([4, 2, 3]);
  });
});

describe('reshape lowering', () => {
  it('reshape [4, 8] -> [32] uses flat index computation', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([32], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [32]).getResult(0)]);
    });

    expect(getLoopExtents(pf.body)).toEqual([32]);

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(1);
    const load = stores[0].value;
    expect(load).toBeInstanceOf(BufferLoadNode);
    expect(load.indices.length).toBe(2);
  });

  it('reshape [6] -> [2, 3] computes multi-dim indices from flat index', () => {
    const inT = new TensorType([6], ScalarType.F32);
    const outT = new TensorType([2, 3], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [2, 3]).getResult(0)]);
    });

    expect(getLoopExtents(pf.body)).toEqual([2, 3]);

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].indices.length).toBe(2);
    expect(stores[0].value.indices.length).toBe(1);
  });

  it('reshape [2, 3, 4] -> [6, 4] uses modulo and floordiv for input indices', () => {
    const inT = new TensorType([2, 3, 4], ScalarType.F32);
    const outT = new TensorType([6, 4], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [6, 4]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    expect(load.indices.length).toBe(3);

    const mathOps = collectNodes(load, n => n instanceof MathOpNode && (n.op === '%' || n.op === '//'));
    expect(mathOps.length).toBeGreaterThan(0);
  });
});

describe('reshape index simplification via mathOp', () => {
  it('reshape [4] -> [4, 1] produces no mod/div — trailing dim=1 simplified away', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4, 1], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [4, 1]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    const modDiv = collectNodes(load, n => n instanceof MathOpNode && (n.op === '%' || n.op === '//'));
    expect(modDiv.length).toBe(0);
  });

  it('reshape [4, 1] -> [4] produces no multiply-by-1 in flat index', () => {
    const inT = new TensorType([4, 1], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [4]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    const mulBy1 = collectNodes(load, n =>
      n instanceof MathOpNode && n.op === '*' &&
      ((n.b instanceof IntImmNode && n.b.value === 1) || (n.a instanceof IntImmNode && n.a.value === 1))
    );
    expect(mulBy1.length).toBe(0);
  });

  it('reshape [1, 6] -> [2, 3] produces no mod-by-1 or div-by-1', () => {
    const inT = new TensorType([1, 6], ScalarType.F32);
    const outT = new TensorType([2, 3], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [2, 3]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    const trivialOps = collectNodes(load, n =>
      n instanceof MathOpNode &&
      ((n.op === '%' && n.b instanceof IntImmNode && n.b.value === 1) ||
       (n.op === '//' && n.b instanceof IntImmNode && n.b.value === 1))
    );
    expect(trivialOps.length).toBe(0);
  });
});

describe('regression — identity reshape uses direct index copy', () => {
  it('same-shape reshape uses no modulo or floordiv', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const outT = new TensorType([4, 8], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [4, 8]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    const modDiv = collectNodes(load, n => n instanceof MathOpNode && (n.op === '%' || n.op === '//'));
    expect(modDiv.length).toBe(0);
  });

  it('identity reshape load indices match store indices directly', () => {
    const inT = new TensorType([3, 5], ScalarType.F32);
    const outT = new TensorType([3, 5], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.reshape(args[0], [3, 5]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const store = stores[0];
    const load = store.value;
    expect(load).toBeInstanceOf(BufferLoadNode);
    expect(load.indices.length).toBe(store.indices.length);
    for (let i = 0; i < load.indices.length; i++) {
      expect(load.indices[i]).toBe(store.indices[i]);
    }
  });
});

describe('slice lowering', () => {
  it('slice with starts=[1, 2] offsets input indices', () => {
    const inT = new TensorType([10, 10], ScalarType.F32);
    const outT = new TensorType([3, 4], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.slice(args[0], [1, 2], [4, 6]).getResult(0)]);
    });

    expect(getLoopExtents(pf.body)).toEqual([3, 4]);

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    expect(load).toBeInstanceOf(BufferLoadNode);

    const adds = collectNodes(load, n => n instanceof MathOpNode && n.op === '+');
    expect(adds.length).toBe(2);

    const offsets = adds.map(a => a.a instanceof IntImmNode ? a.a.value : null).sort();
    expect(offsets).toEqual([1, 2]);
  });

  it('slice with strides=[2, 1] multiplies loop var by stride', () => {
    const inT = new TensorType([10, 10], ScalarType.F32);
    const outT = new TensorType([3, 5], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.slice(args[0], [0, 0], [6, 5], [2, 1]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;

    const muls = collectNodes(load, n => n instanceof MathOpNode && n.op === '*');
    expect(muls.length).toBeGreaterThan(0);
    const strideMul = muls.find(m => m.b instanceof IntImmNode && m.b.value === 2);
    expect(strideMul).toBeDefined();
  });
});

describe('pad lowering', () => {
  it('pad produces IfThenElseNode for bounds checking', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const padValT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([8], ScalarType.F32);
    const pf = lower('f', [inT, padValT], [outT], (b, args) => {
      b.returnOp([b.pad(args[0], args[1], [2], [2]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(IfThenElseNode);

    const compares = collectNodes(stores[0].value.condition, n => n instanceof CompareNode);
    expect(compares.length).toBeGreaterThanOrEqual(2);
  });

  it('pad with interior padding uses modulo check', () => {
    const inT = new TensorType([3], ScalarType.F32);
    const padValT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([9], ScalarType.F32);
    const pf = lower('f', [inT, padValT], [outT], (b, args) => {
      b.returnOp([b.pad(args[0], args[1], [1], [1], [2]).getResult(0)]);
    });

    const mods = collectNodes(pf.body, n => n instanceof MathOpNode && n.op === '%');
    expect(mods.length).toBeGreaterThan(0);
  });
});

describe('concat lowering', () => {
  it('concat on dim 0 produces SeqNode with per-input loop nests', () => {
    const a = new TensorType([3, 4], ScalarType.F32);
    const b_ = new TensorType([5, 4], ScalarType.F32);
    const outT = new TensorType([8, 4], ScalarType.F32);
    const pf = lower('f', [a, b_], [outT], (b, args) => {
      b.returnOp([b.concat([args[0], args[1]], 0).getResult(0)]);
    });

    const blocks = collectNodes(pf.body, n => n instanceof BlockNode);
    expect(blocks.length).toBe(2);
    const names = blocks.map(b => b.name).sort();
    expect(names).toEqual(['concat_0', 'concat_1']);
  });

  it('concat on dim 1 offsets the concat dimension index for second input', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b_ = new TensorType([2, 5], ScalarType.F32);
    const outT = new TensorType([2, 8], ScalarType.F32);
    const pf = lower('f', [a, b_], [outT], (b, args) => {
      b.returnOp([b.concat([args[0], args[1]], 1).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(2);

    const offsetStore = stores.find(s =>
      s.indices.some(idx => idx instanceof MathOpNode && idx.op === '+')
    );
    expect(offsetStore).toBeDefined();
    const addOp = offsetStore.indices.find(idx => idx instanceof MathOpNode && idx.op === '+');
    const offset = addOp.b instanceof IntImmNode ? addOp.b : addOp.a;
    expect(offset).toBeInstanceOf(IntImmNode);
    expect(offset.value).toBe(3);
  });
});

describe('broadcast_in_dim lowering', () => {
  it('broadcast [3] -> [4, 3] with dims=[1] maps input to dim 1', () => {
    const inT = new TensorType([3], ScalarType.F32);
    const outT = new TensorType([4, 3], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.broadcast(args[0], [4, 3], [1]).getResult(0)]);
    });

    expect(getLoopExtents(pf.body)).toEqual([4, 3]);

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const load = stores[0].value;
    expect(load).toBeInstanceOf(BufferLoadNode);
    expect(load.indices.length).toBe(1);
  });

  it('broadcast size-1 dim inserts IntImm(0) index', () => {
    const inT = new TensorType([1, 4], ScalarType.F32);
    const outT = new TensorType([3, 4], ScalarType.F32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.broadcast(args[0], [3, 4], [0, 1]).getResult(0)]);
    });

    const loads = collectNodes(pf.body, n => n instanceof BufferLoadNode);
    const inLoad = loads.find(l => l.buffer.shape.length === 2 && l.buffer.shape[0] === 1);
    expect(inLoad).toBeDefined();
    expect(inLoad.indices[0]).toBeInstanceOf(IntImmNode);
    expect(inLoad.indices[0].value).toBe(0);
  });
});

describe('iota lowering', () => {
  it('iota on dimension 0 casts loop var to output dtype', () => {
    const outT = new TensorType([5, 3], ScalarType.F32);
    const pf = lower('f', [], [outT], (b, args) => {
      b.returnOp([b.iota(0, outT).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(1);
    expect(stores[0].value).toBeInstanceOf(CastNode);
    expect(stores[0].value.toDtype).toBe('f32');
  });

  it('iota on dimension 1 uses second loop variable', () => {
    const outT = new TensorType([3, 5], ScalarType.F32);
    const pf = lower('f', [], [outT], (b, args) => {
      b.returnOp([b.iota(1, outT).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const cast = stores[0].value;
    expect(cast).toBeInstanceOf(CastNode);
    expect(cast.fromDtype).toBe('index');
  });
});

describe('scatter lowering', () => {
  it('scatter produces copy + update with add semantics', () => {
    const operandT = new TensorType([5, 3], ScalarType.F32);
    const indicesT = new TensorType([2, 1], ScalarType.I32);
    const updatesT = new TensorType([2, 3], ScalarType.F32);
    const outT = new TensorType([5, 3], ScalarType.F32);
    const pf = lower('f', [operandT, indicesT, updatesT], [outT], (b, args) => {
      b.returnOp([b.scatter(args[0], args[1], args[2], {
        updateWindowDims: [1],
        insertedWindowDims: [0],
        scatterDimsToOperandDims: [0],
        indexVectorDim: 1,
      }).getResult(0)]);
    });

    expect(pf.body).toBeInstanceOf(SeqNode);
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(2);
    const addStore = stores.find(s => s.value instanceof MathOpNode && s.value.op === '+');
    expect(addStore).toBeDefined();
    expect(addStore.value.a).toBeInstanceOf(BufferLoadNode);
    expect(addStore.value.b).toBeInstanceOf(BufferLoadNode);
  });

  it('scatterAdd is an alias for scatter', () => {
    const operandT = new TensorType([4], ScalarType.F32);
    const indicesT = new TensorType([2, 1], ScalarType.I32);
    const updatesT = new TensorType([2], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [operandT, indicesT, updatesT], [outT], (b, args) => {
      b.returnOp([b.scatterAdd(args[0], args[1], args[2], {
        updateWindowDims: [],
        insertedWindowDims: [0],
        scatterDimsToOperandDims: [0],
        indexVectorDim: 1,
      }).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const addStore = stores.find(s => s.value instanceof MathOpNode && s.value.op === '+');
    expect(addStore).toBeDefined();
  });
});
