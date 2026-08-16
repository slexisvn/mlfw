import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../../src/compiler/passes/lowering/graph_to_tensor.js';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, FloatImmNode, IntImmNode, SeqNode,
  CallExternNode, CompareNode, IfThenElseNode, AllocateNode
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

function getSeqStmts(node) {
  if (node instanceof SeqNode) return node.stmts;
  return [node];
}

describe('reduce sum lowering', () => {
  it('sum over last dim produces init + accumulate phases', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const initT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.F32);
    const pf = lower('f', [inT, initT], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'sum').getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    expect(stmts.length).toBe(2);

    const initBlocks = collectNodes(stmts[0], n => n instanceof BlockNode);
    expect(initBlocks.length).toBe(1);
    expect(initBlocks[0].name).toMatch(/^reduce_init_\d+$/);

    const accBlocks = collectNodes(stmts[1], n => n instanceof BlockNode);
    expect(accBlocks.length).toBe(1);
    expect(accBlocks[0].name).toMatch(/^reduce_acc_\d+$/);

    const accStores = collectNodes(stmts[1], n => n instanceof BufferStoreNode);
    expect(accStores[0].value).toBeInstanceOf(MathOpNode);
    expect(accStores[0].value.op).toBe('+');
  });

  it('reduce all dims of 2D tensor to scalar produces no spatial loops in init', () => {
    const inT = new TensorType([3, 4], ScalarType.F32);
    const initT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([], ScalarType.F32);
    const pf = lower('f', [inT, initT], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [0, 1], 'sum').getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const initFors = collectNodes(stmts[0], n => n instanceof ForNode);
    expect(initFors.length).toBe(0);
  });
});

describe('reduce mean lowering', () => {
  it('mean produces init + accumulate + division phases', () => {
    const inT = new TensorType([6, 4], ScalarType.F32);
    const initT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([6], ScalarType.F32);
    const pf = lower('f', [inT, initT], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'mean').getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    expect(stmts.length).toBe(3);

    const divBlocks = collectNodes(stmts[2], n => n instanceof BlockNode);
    expect(divBlocks[0].name).toMatch(/^mean_div_\d+$/);

    const divStores = collectNodes(stmts[2], n => n instanceof BufferStoreNode);
    const mulNode = divStores[0].value;
    expect(mulNode).toBeInstanceOf(MathOpNode);
    expect(mulNode.op).toBe('*');

    const floats = collectNodes(mulNode, n => n instanceof FloatImmNode);
    const divFactor = floats.find(f => Math.abs(f.value - 1 / 4) < 1e-6);
    expect(divFactor).toBeDefined();
  });
});

describe('reduce max lowering', () => {
  it('max reduce uses CallExternNode("max") as combiner', () => {
    const inT = new TensorType([3, 5], ScalarType.F32);
    const initT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([3], ScalarType.F32);
    const pf = lower('f', [inT, initT], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'max').getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const accStores = collectNodes(stmts[1], n => n instanceof BufferStoreNode);
    const combiner = accStores[0].value;
    expect(combiner).toBeInstanceOf(CallExternNode);
    expect(combiner.externName).toBe('max');
  });
});

describe('reduce prod lowering', () => {
  it('prod reduce uses MathOpNode(*) as combiner', () => {
    const inT = new TensorType([2, 3], ScalarType.F32);
    const initT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([2], ScalarType.F32);
    const pf = lower('f', [inT, initT], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'prod').getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const accStores = collectNodes(stmts[1], n => n instanceof BufferStoreNode);
    expect(accStores[0].value).toBeInstanceOf(MathOpNode);
    expect(accStores[0].value.op).toBe('*');
  });
});

describe('argmax lowering', () => {
  it('argmax keeps its running best in a loop-private scratch and compares with gt', () => {
    const inT = new TensorType([3, 5], ScalarType.F32);
    const outT = new TensorType([3], ScalarType.I32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.argmax(args[0], 1).getResult(0)]);
    });

    const allocs = collectNodes(pf.body, n => n instanceof AllocateNode);
    expect(allocs.length).toBe(1);
    expect(allocs[0].scope).toBe('local');
    expect(allocs[0].buffer.shape).toEqual([1]);

    const initBlocks = collectNodes(pf.body, n => n instanceof BlockNode && /^arg_init_\d+$/.test(n.name));
    expect(initBlocks.length).toBe(1);
    const initInsideAlloc = collectNodes(allocs[0].body, n => n === initBlocks[0]);
    expect(initInsideAlloc.length).toBe(1);

    const negInfStore = collectNodes(pf.body, n => n instanceof BufferStoreNode).find(s =>
      s.value instanceof FloatImmNode && s.value.value === -Infinity
    );
    expect(negInfStore).toBeDefined();
    expect(negInfStore.buffer.name).toBe(allocs[0].buffer.name);

    const gtCompare = collectNodes(pf.body, n => n instanceof CompareNode).find(c => c.direction === 'gt');
    expect(gtCompare).toBeDefined();

    const ites = collectNodes(pf.body, n => n instanceof IfThenElseNode);
    expect(ites.length).toBeGreaterThanOrEqual(2);
  });
});

describe('argmin lowering', () => {
  it('argmin initializes with +Infinity and uses lt comparison', () => {
    const inT = new TensorType([4, 6], ScalarType.F32);
    const outT = new TensorType([4], ScalarType.I32);
    const pf = lower('f', [inT], [outT], (b, args) => {
      b.returnOp([b.argmin(args[0], 1).getResult(0)]);
    });

    const posInfStore = collectNodes(pf.body, n => n instanceof BufferStoreNode).find(s =>
      s.value instanceof FloatImmNode && s.value.value === Infinity
    );
    expect(posInfStore).toBeDefined();

    const ltCompare = collectNodes(pf.body, n => n instanceof CompareNode).find(c => c.direction === 'lt');
    expect(ltCompare).toBeDefined();
  });
});

describe('reduce on first dim', () => {
  it('sum over dim 0 of [4, 8] produces output with shape [8]', () => {
    const inT = new TensorType([4, 8], ScalarType.F32);
    const initT = new TensorType([], ScalarType.F32);
    const outT = new TensorType([8], ScalarType.F32);
    const pf = lower('f', [inT, initT], [outT], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [0], 'sum').getResult(0)]);
    });

    const stores = collectNodes(pf.body, n => n instanceof BufferStoreNode);
    const outStore = stores.find(s => s.buffer.shape.length === 1 && s.buffer.shape[0] === 8);
    expect(outStore).toBeDefined();
  });
});
