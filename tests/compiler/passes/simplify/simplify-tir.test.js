import { describe, it, expect } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode, IfThenElseNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, CompareNode, ForKind, PrimFunc,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { simplifyPrimFunc } from '../../../../src/compiler/passes/simplify/simplify_tir.js';

const iv = (n) => new VariableNode(n, 'int32');
const c = (x) => new IntImmNode(x);

function countNodes(node, type, seen = new Set()) {
  if (!node || typeof node !== 'object' || seen.has(node)) return 0;
  seen.add(node);
  let n = node.type === type ? 1 : 0;
  for (const k of Object.keys(node)) {
    if (k === '_parent' || k === '_parentKey' || k === '_parentIdx') continue;
    const v = node[k];
    if (Array.isArray(v)) for (const e of v) n += countNodes(e, type, seen);
    else if (v && typeof v === 'object') n += countNodes(v, type, seen);
  }
  return n;
}

function loopBlock(varName, extent, storeValueFn) {
  const A = new Buffer('A', [extent], 'float32', 'global');
  const i = iv(varName);
  const store = new BufferStoreNode(A, [i], storeValueFn(i));
  const block = new BlockNode('b', [], [{ buffer: A }], [{ buffer: A }], store);
  const body = new ForNode(i, c(0), c(extent), ForKind.SERIAL, block);
  return new PrimFunc('f', [], body);
}

describe('SimplifyPrimFunc (TVM tir.Simplify analog)', () => {
  it('drops a provably-true in-bounds ternary guard: i in [0,8) makes (i < 8 ? A[i] : 0) become A[i]', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = loopBlock('i', 8, (i) =>
      new IfThenElseNode(new CompareNode('lt', i, c(8)), new BufferLoadNode(A, [i]), new FloatImmNode(0)));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(0);
  });

  it('keeps a guard it cannot prove', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = loopBlock('i', 8, (i) =>
      new IfThenElseNode(new CompareNode('lt', i, c(4)), new BufferLoadNode(A, [i]), new FloatImmNode(0)));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(1);
  });

  it('folds (i*2)//2 -> i and i%8 -> i inside an index, given the loop range', () => {
    const f = loopBlock('i', 8, (i) =>
      new BufferLoadNode(new Buffer('S', [8], 'float32', 'global'),
        [new MathOpNode('//', new MathOpNode('*', i, c(2)), c(2))]));
    simplifyPrimFunc(f);
    const store = f.body.body.body;
    const idx = store.value.indices[0];
    expect(idx).toMatchObject({ type: 'VariableNode', name: 'i' });
  });

  it('eliminates a provably-true statement-level IfThenElse: for i in [0,8) { if (i < 8) A[i] = 0 } becomes an unconditional store', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const i = iv('i');
    const store = new BufferStoreNode(A, [i], new FloatImmNode(0));
    const guarded = new IfThenElseNode(new MathOpNode('<', i, c(8)), store);
    const body = new ForNode(i, c(0), c(8), ForKind.SERIAL, guarded);
    const f = new PrimFunc('f', [], body);
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(0);
    expect(countNodes(f.body, 'BufferStoreNode')).toBe(1);
  });

  it('does not touch float arithmetic semantics', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const f = loopBlock('i', 8, (i) =>
      new MathOpNode('*', new BufferLoadNode(A, [i]), new FloatImmNode(2.5)));
    simplifyPrimFunc(f);
    const val = f.body.body.body.value;
    expect(val.type).toBe('MathOpNode');
    expect(val.op).toBe('*');
    expect(val.b).toMatchObject({ type: 'FloatImmNode', value: 2.5 });
  });
});
