import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType, DYNAMIC } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import {
  LoweringContext, hasLoweringRule, makeLoopNest, wrapLoops,
  computeBroadcastIndices
} from '../../../../src/compiler/passes/lowering/lowering_registry.js';
import {
  ForNode, BufferStoreNode, BufferLoadNode, IntImmNode, VariableNode, FloatImmNode
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';

const SKIP_KEYS = new Set(['_parent', '_parentKey', '_parentIdx']);

function collectNodesLocal(node, predicate) {
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

describe('LoweringContext', () => {
  it('allocVar produces unique names with incrementing counter', () => {
    const ctx = new LoweringContext();
    const v1 = ctx.allocVar('x');
    const v2 = ctx.allocVar('x');
    expect(v1.name).toBe('x_0');
    expect(v2.name).toBe('x_1');
    expect(v1).not.toBe(v2);
  });

  it('allocVar returns VariableNode with default int32 dtype', () => {
    const ctx = new LoweringContext();
    const v = ctx.allocVar('test');
    expect(v).toBeInstanceOf(VariableNode);
    expect(v.dtype).toBe('int32');
  });

  it('allocVar respects custom dtype', () => {
    const ctx = new LoweringContext();
    const v = ctx.allocVar('f', 'f32');
    expect(v.dtype).toBe('f32');
  });

  it('getOrAllocBuffer returns same buffer for same value', () => {
    const ctx = new LoweringContext();
    const mockValue = { type: { shape: [4, 8], dtype: 'f32' } };
    const buf1 = ctx.getOrAllocBuffer(mockValue);
    const buf2 = ctx.getOrAllocBuffer(mockValue);
    expect(buf1).toBe(buf2);
  });

  it('getOrAllocBuffer creates buffer with correct shape and dtype', () => {
    const ctx = new LoweringContext();
    const mockValue = { type: { shape: [3, 5], dtype: 'f16' } };
    const buf = ctx.getOrAllocBuffer(mockValue);
    expect(buf.shape).toEqual([3, 5]);
    expect(buf.dtype).toBe('f16');
  });

  it('extentNode returns IntImmNode for static dims', () => {
    const ctx = new LoweringContext();
    const buf = new Buffer('test', [4], 'f32', 'global');
    const node = ctx.extentNode(4, buf);
    expect(node).toBeInstanceOf(IntImmNode);
    expect(node.value).toBe(4);
  });

  it('extentNode returns VariableNode for DYNAMIC dims', () => {
    const ctx = new LoweringContext();
    const buf = new Buffer('test', [DYNAMIC], 'f32', 'global');
    const node = ctx.extentNode(DYNAMIC, buf, 0);
    expect(node).toBeInstanceOf(VariableNode);
  });

  it('extentNode reuses same VariableNode for repeated DYNAMIC access', () => {
    const ctx = new LoweringContext();
    const buf = new Buffer('test', [DYNAMIC], 'f32', 'global');
    const n1 = ctx.extentNode(DYNAMIC, buf, 0);
    const n2 = ctx.extentNode(DYNAMIC, buf, 0);
    expect(n1).toBe(n2);
  });

  it('allocVarArray creates array of unique variables', () => {
    const ctx = new LoweringContext();
    const vars = ctx.allocVarArray('loop', 3);
    expect(vars.length).toBe(3);
    const names = vars.map(v => v.name);
    expect(new Set(names).size).toBe(3);
  });
});

describe('hasLoweringRule', () => {
  it('returns true for registered elementwise ops', () => {
    expect(hasLoweringRule('add')).toBe(true);
    expect(hasLoweringRule('mul')).toBe(true);
    expect(hasLoweringRule('exp')).toBe(true);
  });

  it('returns true for constant ops', () => {
    expect(hasLoweringRule('constant')).toBe(true);
    expect(hasLoweringRule('scalar_constant')).toBe(true);
  });

  it('returns true for shape ops', () => {
    expect(hasLoweringRule('transpose')).toBe(true);
    expect(hasLoweringRule('reshape')).toBe(true);
    expect(hasLoweringRule('slice')).toBe(true);
  });

  it('returns false for unknown ops', () => {
    expect(hasLoweringRule('nonexistent_op_xyz')).toBe(false);
  });
});

describe('makeLoopNest', () => {
  it('creates matching loopVars and loopBinds arrays', () => {
    const ctx = new LoweringContext();
    const { loopVars, loopBinds, indices } = makeLoopNest(ctx, [4, 8]);
    expect(loopVars.length).toBe(2);
    expect(loopBinds.length).toBe(2);
    expect(indices.length).toBe(2);
  });
});

describe('wrapLoops', () => {
  it('creates nested ForNode chain with correct extents', () => {
    const ctx = new LoweringContext();
    const loopVars = ctx.allocVarArray('i', 3);
    const body = new BufferStoreNode(null, [], null);

    const wrapped = wrapLoops(body, loopVars, [2, 3, 4]);

    expect(wrapped).toBeInstanceOf(ForNode);
    expect(wrapped.extent.value).toBe(2);
    expect(wrapped.body).toBeInstanceOf(ForNode);
    expect(wrapped.body.extent.value).toBe(3);
    expect(wrapped.body.body).toBeInstanceOf(ForNode);
    expect(wrapped.body.body.extent.value).toBe(4);
  });
});

describe('computeBroadcastIndices', () => {
  it('maps size-1 dims to IntImm(0)', () => {
    const inBuf = new Buffer('in', [1, 4], 'f32', 'global');
    const outBuf = new Buffer('out', [3, 4], 'f32', 'global');
    const outIndices = [new VariableNode('i0', 'int32'), new VariableNode('i1', 'int32')];
    const result = computeBroadcastIndices(inBuf, outBuf, outIndices);

    expect(result.length).toBe(2);
    expect(result[0]).toBeInstanceOf(IntImmNode);
    expect(result[0].value).toBe(0);
    expect(result[1]).toBe(outIndices[1]);
  });

  it('handles rank difference by offsetting from output rank', () => {
    const inBuf = new Buffer('in', [4], 'f32', 'global');
    const outBuf = new Buffer('out', [3, 4], 'f32', 'global');
    const outIndices = [new VariableNode('i0', 'int32'), new VariableNode('i1', 'int32')];
    const result = computeBroadcastIndices(inBuf, outBuf, outIndices);

    expect(result.length).toBe(1);
    expect(result[0]).toBe(outIndices[1]);
  });
});

describe('constant lowering', () => {
  it('scalar constant lowers to a single BufferStoreNode with no loops', () => {
    const scalarT = new TensorType([], ScalarType.F32);
    const pf = lowerGraphToPrimFunc(
      buildFunction('f', [], [scalarT], (b, args) => {
        b.returnOp([b.scalarConstant(42, ScalarType.F32).getResult(0)]);
      })
    );

    const stores = collectNodesLocal(pf.body, n => n instanceof BufferStoreNode);

    expect(stores.length).toBe(1);
    expect(stores[0].value).toBeInstanceOf(FloatImmNode);
    expect(stores[0].value.value).toBe(42);
    expect(stores[0].indices.length).toBe(0);
  });

  it('tensor constant lowers to a loop nest storing the value everywhere', () => {
    const t = new TensorType([2, 3], ScalarType.F32);
    const pf = lowerGraphToPrimFunc(
      buildFunction('f', [], [t], (b, args) => {
        b.returnOp([b.tensorConstant(7, [2, 3], ScalarType.F32).getResult(0)]);
      })
    );

    const fors = [];
    let cur = pf.body;
    while (cur instanceof ForNode) {
      fors.push(cur);
      cur = cur.body;
    }
    expect(fors.length).toBe(2);
    expect(fors[0].extent.value).toBe(2);
    expect(fors[1].extent.value).toBe(3);
  });
});

describe('lowerGraphToPrimFunc output structure', () => {
  it('PrimFunc params include arg vars and ret vars', () => {
    const inT = new TensorType([4], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const pf = lowerGraphToPrimFunc(
      buildFunction('f', [inT], [outT], (b, args) => {
        b.returnOp([b.neg(args[0]).getResult(0)]);
      })
    );

    expect(pf.params.length).toBeGreaterThanOrEqual(2);
    expect(pf.bufferMap.size).toBeGreaterThanOrEqual(2);
  });

  it('throws for ops with no lowering rule', () => {
    const t = new TensorType([4], ScalarType.F32);
    const func = buildFunction('f', [t], [t], (b, args) => {
      const op = b._buildOp('totally_fake_op', [args[0]], [t]);
      b.returnOp([op.getResult(0)]);
    });

    expect(() => lowerGraphToPrimFunc(func)).toThrow(/No lowering rule/);
  });
});
