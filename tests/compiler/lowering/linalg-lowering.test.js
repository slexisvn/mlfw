import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, FloatImmNode, IntImmNode, SeqNode,
  CompareNode, IfThenElseNode, CallExternNode
} from '../../../src/compiler/ir/tensor/nodes.js';

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

describe('matmul (dot) lowering', () => {
  it('matmul [4, 8] x [8, 6] produces init + accumulate with multiply-add', () => {
    const lhsT = new TensorType([4, 8], ScalarType.F32);
    const rhsT = new TensorType([8, 6], ScalarType.F32);
    const outT = new TensorType([4, 6], ScalarType.F32);
    const pf = lower('f', [lhsT, rhsT], [outT], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    expect(stmts.length).toBe(2);

    const initBlocks = collectNodes(stmts[0], n => n instanceof BlockNode);
    expect(initBlocks[0].name).toMatch(/^matmul_init_\d+$/);

    const initStores = collectNodes(stmts[0], n => n instanceof BufferStoreNode);
    expect(initStores[0].value).toBeInstanceOf(FloatImmNode);
    expect(initStores[0].value.value).toBe(0);

    const accBlocks = collectNodes(stmts[1], n => n instanceof BlockNode);
    expect(accBlocks[0].name).toMatch(/^matmul_\d+$/);

    const accStores = collectNodes(stmts[1], n => n instanceof BufferStoreNode);
    const accExpr = accStores[0].value;
    expect(accExpr).toBeInstanceOf(MathOpNode);
    expect(accExpr.op).toBe('+');
    expect(accExpr.b).toBeInstanceOf(MathOpNode);
    expect(accExpr.b.op).toBe('*');
  });

  it('matmul accumulate has 3 loop vars: lhs_spatial, rhs_spatial, contracting', () => {
    const lhsT = new TensorType([4, 8], ScalarType.F32);
    const rhsT = new TensorType([8, 6], ScalarType.F32);
    const outT = new TensorType([4, 6], ScalarType.F32);
    const pf = lower('f', [lhsT, rhsT], [outT], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const accFors = collectNodes(stmts[1], n => n instanceof ForNode);
    expect(accFors.length).toBe(3);

    const extents = accFors.map(f => f.extent.value).sort((a, b) => a - b);
    expect(extents).toEqual([4, 6, 8]);
  });
});

describe('batched matmul lowering', () => {
  it('batched matmul [2, 4, 8] x [2, 8, 6] adds batch loop var', () => {
    const lhsT = new TensorType([2, 4, 8], ScalarType.F32);
    const rhsT = new TensorType([2, 8, 6], ScalarType.F32);
    const outT = new TensorType([2, 4, 6], ScalarType.F32);
    const pf = lower('f', [lhsT, rhsT], [outT], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const accFors = collectNodes(stmts[1], n => n instanceof ForNode);
    expect(accFors.length).toBe(4);

    const extents = accFors.map(f => f.extent.value).sort((a, b) => a - b);
    expect(extents).toEqual([2, 4, 6, 8]);
  });
});

describe('conv lowering', () => {
  it('conv2d NCHW produces init + 6-deep loop nest for accumulation', () => {
    const inT = new TensorType([1, 3, 8, 8], ScalarType.F32);
    const kerT = new TensorType([16, 3, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 16, 6, 6], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    expect(stmts.length).toBe(2);

    const initBlocks = collectNodes(stmts[0], n => n instanceof BlockNode);
    expect(initBlocks[0].name).toMatch(/^conv_init_\d+$/);

    const accBlocks = collectNodes(stmts[1], n => n instanceof BlockNode);
    expect(accBlocks[0].name).toMatch(/^conv_acc_\d+$/);

    const accFors = collectNodes(stmts[1], n => n instanceof ForNode);
    expect(accFors.length).toBeGreaterThanOrEqual(6);
  });

  it('conv2d with padding generates bounds-check guards', () => {
    const inT = new TensorType([1, 1, 5, 5], ScalarType.F32);
    const kerT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 1, 5, 5], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const compares = collectNodes(stmts[1], n => n instanceof CompareNode);
    expect(compares.length).toBeGreaterThanOrEqual(4);

    const ites = collectNodes(stmts[1], n => n instanceof IfThenElseNode);
    expect(ites.length).toBe(1);
  });

  it('conv2d accumulate expression is out + guarded(in * kernel)', () => {
    const inT = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const kerT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const accStores = collectNodes(stmts[1], n => n instanceof BufferStoreNode);
    const expr = accStores[0].value;
    expect(expr).toBeInstanceOf(MathOpNode);
    expect(expr.op).toBe('+');

    expect(expr.a).toBeInstanceOf(BufferLoadNode);

    const mulNode = expr.b instanceof IfThenElseNode ? expr.b.condition : expr.b;
    expect(expr.b instanceof MathOpNode || expr.b instanceof IfThenElseNode).toBe(true);
  });

  it('conv2d with stride=2 embeds stride multiplication in spatial index', () => {
    const inT = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const kerT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const muls = collectNodes(stmts[1], n => n instanceof MathOpNode && n.op === '*');
    const strideMuls = muls.filter(m => m.b instanceof IntImmNode && m.b.value === 2);
    expect(strideMuls.length).toBeGreaterThan(0);
  });
});

describe('conv index simplification via mathOp', () => {
  it('stride=1 dilation=1 pad=0 produces no multiply-by-1 or add-zero', () => {
    const inT = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const kerT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 1, 2, 2], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const allMath = collectNodes(stmts[1], n => n instanceof MathOpNode);
    const mulBy1 = allMath.filter(n =>
      n.op === '*' &&
      ((n.b instanceof IntImmNode && n.b.value === 1) || (n.a instanceof IntImmNode && n.a.value === 1))
    );
    expect(mulBy1.length).toBe(0);

    const addZero = allMath.filter(n =>
      (n.op === '+' || n.op === '-') &&
      ((n.b instanceof IntImmNode && n.b.value === 0) || (n.a instanceof IntImmNode && n.a.value === 0))
    );
    expect(addZero.length).toBe(0);
  });

  it('stride=2 keeps stride multiplication, still no multiply-by-1', () => {
    const inT = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const kerT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [2, 2], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const allMath = collectNodes(stmts[1], n => n instanceof MathOpNode);
    const strideMul = allMath.filter(n =>
      n.op === '*' && n.b instanceof IntImmNode && n.b.value === 2
    );
    expect(strideMul.length).toBeGreaterThan(0);

    const mulBy1 = allMath.filter(n =>
      n.op === '*' &&
      ((n.b instanceof IntImmNode && n.b.value === 1) || (n.a instanceof IntImmNode && n.a.value === 1))
    );
    expect(mulBy1.length).toBe(0);
  });
});

describe('regression — conv pad=0 emits no bounds checks', () => {
  it('conv with zero padding produces no CompareNode or IfThenElseNode', () => {
    const inT = new TensorType([1, 2, 6, 6], ScalarType.F32);
    const kerT = new TensorType([4, 2, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 4, 4, 4], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const compares = collectNodes(stmts[1], n => n instanceof CompareNode);
    expect(compares.length).toBe(0);

    const ites = collectNodes(stmts[1], n => n instanceof IfThenElseNode);
    expect(ites.length).toBe(0);
  });

  it('conv with asymmetric padding still has bounds checks', () => {
    const inT = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const kerT = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const outT = new TensorType([1, 1, 4, 4], ScalarType.F32);
    const pf = lower('f', [inT, kerT], [outT], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const compares = collectNodes(stmts[1], n => n instanceof CompareNode);
    expect(compares.length).toBeGreaterThanOrEqual(4);
  });
});

describe('dot with explicit contracting dims', () => {
  it('dot with custom contracting dims produces correct geometry', () => {
    const lhsT = new TensorType([3, 5], ScalarType.F32);
    const rhsT = new TensorType([7, 5], ScalarType.F32);
    const outT = new TensorType([3, 7], ScalarType.F32);
    const pf = lower('f', [lhsT, rhsT], [outT], (b, args) => {
      b.returnOp([b.dot(args[0], args[1], [1], [1]).getResult(0)]);
    });

    const stmts = getSeqStmts(pf.body);
    const accFors = collectNodes(stmts[1], n => n instanceof ForNode);
    const extents = accFors.map(f => f.extent.value).sort((a, b) => a - b);
    expect(extents).toEqual([3, 5, 7]);
  });
});
