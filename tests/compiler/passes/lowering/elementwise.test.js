import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { CPUTarget, WasmTarget } from '../../../../src/compiler/support/target.js';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, FloatImmNode, IntImmNode, CallExternNode, CompareNode,
  IfThenElseNode, CastNode, SeqNode
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

function findForDepth(node) {
  let depth = 0;
  let cur = node;
  while (cur instanceof ForNode) {
    depth++;
    cur = cur.body;
  }
  return depth;
}

describe('binary arithmetic lowering', () => {
  it('add lowers to a loop nest with MathOpNode(+)', () => {
    const t = new TensorType([4, 8], ScalarType.F32);
    const pf = lower('f', [t, t], [t], (b, args) => {
      const r = b.add(args[0], args[1]);
      b.returnOp([r.getResult(0)]);
    });

    expect(pf.name).toBe('f');
    expect(pf.params.length).toBeGreaterThanOrEqual(3);
    expect(findForDepth(pf.body)).toBe(2);

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(1);
    expect(stores[0].value).toBeInstanceOf(MathOpNode);
    expect(stores[0].value.op).toBe('+');
    expect(stores[0].value.a).toBeInstanceOf(BufferLoadNode);
    expect(stores[0].value.b).toBeInstanceOf(BufferLoadNode);
  });

  it('sub lowers to MathOpNode(-)', () => {
    const t = new TensorType([3], ScalarType.F32);
    const pf = lower('f', [t, t], [t], (b, args) => {
      b.returnOp([b.sub(args[0], args[1]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value.op).toBe('-');
  });

  it('mul lowers to MathOpNode(*)', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const pf = lower('f', [t, t], [t], (b, args) => {
      b.returnOp([b.mul(args[0], args[1]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value.op).toBe('*');
  });

  it('div lowers to MathOpNode(/)', () => {
    const t = new TensorType([5], ScalarType.F32);
    const pf = lower('f', [t, t], [t], (b, args) => {
      b.returnOp([b.div(args[0], args[1]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value.op).toBe('/');
  });
});

describe('unary elementwise lowering', () => {
  it('neg lowers to unary -x', () => {
    const t = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(1);
    const val = stores[0].value;
    expect(val).toBeInstanceOf(MathOpNode);
    expect(val.op).toBe('-');
    expect(val.b).toBeFalsy();
    expect(val.a).toBeInstanceOf(BufferLoadNode);
  });

  it('exp lowers to CallExternNode("exp")', () => {
    const t = new TensorType([3, 2], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(CallExternNode);
    expect(stores[0].value.externName).toBe('exp');
    expect(stores[0].value.args.length).toBe(1);
  });

  it('sqrt lowers to CallExternNode("sqrt")', () => {
    const t = new TensorType([6], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.sqrt(args[0]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value.externName).toBe('sqrt');
  });

  it('tanh lowers to CallExternNode("tanh")', () => {
    const t = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.tanh(args[0]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value.externName).toBe('tanh');
  });
});

describe('compare lowering', () => {
  it('compare lowers to CompareNode with correct direction', () => {
    const t = new TensorType([4], ScalarType.F32);
    const boolT = new TensorType([4], ScalarType.BOOL);
    const pf = lower('f', [t, t], [boolT], (b, args) => {
      b.returnOp([b.compare(args[0], args[1], 'lt').getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(CompareNode);
    expect(stores[0].value.direction).toBe('lt');
  });
});

describe('select lowering', () => {
  it('select lowers to IfThenElseNode', () => {
    const t = new TensorType([4], ScalarType.F32);
    const boolT = new TensorType([4], ScalarType.BOOL);
    const pf = lower('f', [boolT, t, t], [t], (b, args) => {
      b.returnOp([b.select(args[0], args[1], args[2]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(IfThenElseNode);
    expect(stores[0].value.condition).toBeInstanceOf(BufferLoadNode);
    expect(stores[0].value.thenBody).toBeInstanceOf(BufferLoadNode);
    expect(stores[0].value.elseBody).toBeInstanceOf(BufferLoadNode);
  });
});

describe('clamp lowering', () => {
  it('clamp lowers to nested min(max(x, lo), hi)', () => {
    const t = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [t, t, t], [t], (b, args) => {
      b.returnOp([b.clamp(args[0], args[1], args[2]).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const val = stores[0].value;
    expect(val).toBeInstanceOf(CallExternNode);
    expect(val.externName).toBe('min');
    expect(val.args[0]).toBeInstanceOf(CallExternNode);
    expect(val.args[0].externName).toBe('max');
  });
});

describe('convert lowering', () => {
  it('convert lowers to CastNode', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.I32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.convert(args[0], ScalarType.I32).getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(CastNode);
    expect(stores[0].value.fromDtype).toBe('f32');
    expect(stores[0].value.toDtype).toBe('i32');
  });
});

describe('broadcast-aware indexing', () => {
  it('add with broadcast inserts IntImm(0) for size-1 dims', () => {
    const a = new TensorType([4, 8], ScalarType.F32);
    const b_ = new TensorType([1, 8], ScalarType.F32);
    const out = new TensorType([4, 8], ScalarType.F32);
    const pf = lower('f', [a, b_], [out], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });

    const loads = collectNodes(pf.body, n => n instanceof BufferLoadNode);
    const broadcastLoad = loads.find(l => l.buffer.shape[0] === 1);
    expect(broadcastLoad).toBeDefined();
    expect(broadcastLoad.indices[0]).toBeInstanceOf(IntImmNode);
    expect(broadcastLoad.indices[0].value).toBe(0);
  });
});

describe('chained elementwise lowering', () => {
  it('neg(add(x, y)) produces two loop nests in sequence', () => {
    const t = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [t, t], [t], (b, args) => {
      const sum = b.add(args[0], args[1]);
      const negated = b.neg(sum.getResult(0));
      b.returnOp([negated.getResult(0)]);
    });

    const fors = collectNodes(pf.body, n => n instanceof ForNode);
    expect(fors.length).toBeGreaterThanOrEqual(2);

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores.length).toBe(2);

    const addStore = stores.find(s => s.value instanceof MathOpNode && s.value.op === '+');
    const negStore = stores.find(s => s.value instanceof MathOpNode && s.value.op === '-');
    expect(addStore).toBeDefined();
    expect(negStore).toBeDefined();
  });
});

describe('new primitive unary lowering', () => {
  const ops = [
    ['erf', 'erf', CallExternNode],
    ['log2', 'log2', CallExternNode],
    ['log10', 'log10', CallExternNode],
    ['exp2', 'exp2', CallExternNode],
  ];

  for (const [opName, externName, nodeType] of ops) {
    it(`${opName} lowers to CallExternNode("${externName}")`, () => {
      const t = new TensorType([4], ScalarType.F32);
      const pf = lower('f', [t], [t], (b, args) => {
        b.returnOp([b[opName](args[0]).getResult(0)]);
      });
      const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
      expect(stores[0].value).toBeInstanceOf(nodeType);
      expect(stores[0].value.externName).toBe(externName);
    });
  }

  it('square lowers to MathOpNode(*) with same operand on both sides', () => {
    const t = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.square(args[0]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(MathOpNode);
    expect(stores[0].value.op).toBe('*');
    expect(stores[0].value.a).toBeInstanceOf(BufferLoadNode);
    expect(stores[0].value.b).toBeInstanceOf(BufferLoadNode);
  });

  it('reciprocal lowers to MathOpNode(/) with 1 as numerator', () => {
    const t = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.reciprocal(args[0]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(MathOpNode);
    expect(stores[0].value.op).toBe('/');
    expect(stores[0].value.a).toBeInstanceOf(FloatImmNode);
    expect(stores[0].value.a.value).toBe(1);
  });
});

describe('boolean ops lowering', () => {
  it('logical_not lowers to MathOpNode(!)', () => {
    const boolT = new TensorType([4], ScalarType.BOOL);
    const pf = lower('f', [boolT], [boolT], (b, args) => {
      b.returnOp([b.logicalNot(args[0]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(MathOpNode);
    expect(stores[0].value.op).toBe('!');
  });

  it('logical_and lowers to MathOpNode(&&)', () => {
    const boolT = new TensorType([4], ScalarType.BOOL);
    const pf = lower('f', [boolT, boolT], [boolT], (b, args) => {
      b.returnOp([b.logicalAnd(args[0], args[1]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(MathOpNode);
    expect(stores[0].value.op).toBe('&&');
  });

  it('logical_or lowers to MathOpNode(||)', () => {
    const boolT = new TensorType([4], ScalarType.BOOL);
    const pf = lower('f', [boolT, boolT], [boolT], (b, args) => {
      b.returnOp([b.logicalOr(args[0], args[1]).getResult(0)]);
    });
    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    expect(stores[0].value).toBeInstanceOf(MathOpNode);
    expect(stores[0].value.op).toBe('||');
  });
});

describe('loop extents match tensor shape', () => {
  it('3D tensor produces 3-deep loop nest with correct extents', () => {
    const t = new TensorType([2, 3, 4], ScalarType.F32);
    const pf = lower('f', [t], [t], (b, args) => {
      b.returnOp([b.neg(args[0]).getResult(0)]);
    });

    let cur = pf.body;
    const extents = [];
    while (cur instanceof ForNode) {
      expect(cur.extent.value).toBeDefined();
      extents.push(cur.extent.value);
      cur = cur.body;
    }
    expect(extents).toEqual([2, 3, 4]);
  });
});

describe('array-valued constants materialize their data (regression: used to lower to 0)', () => {
  const F = ScalarType.F32, I8 = ScalarType.I8;
  const T = (s, d = F) => new TensorType(s, d);

  for (const [tname, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`f32 array constant feeding add is correct on ${tname}`, () => {
      const data = [10, 20, 30, 40, 50, 60];
      const func = buildFunction('ac', [T([2, 3])], [T([2, 3])], (b, a) => {
        const c = b.constant(data, T([2, 3], F));
        b.returnOp([b.add(a[0], c.getResult(0)).getResult(0)]);
      });
      const out = new Float32Array(6);
      compileGraph(func, makeTarget(), {}).run('ac', new Float32Array(6).fill(1), out);
      for (let i = 0; i < 6; i++) expect(out[i]).toBeCloseTo(data[i] + 1, 5);
    });

    it(`int8 array constant materializes exact values on ${tname}`, () => {
      const data = [1, -2, 3, -4];
      const func = buildFunction('ic', [T([1])], [T([2, 2])], (b) => {
        const c = b.constant(data, T([2, 2], I8));
        b.returnOp([b._buildOp('convert', [c.getResult(0)], [T([2, 2], F)], { target_dtype: F }).getResult(0)]);
      });
      const out = new Float32Array(4);
      compileGraph(func, makeTarget(), {}).run('ic', new Float32Array([0]), out);
      expect([...out]).toEqual(data);
    });
  }
});
