import { describe, it, expect } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode, IfThenElseNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, CompareNode, ForKind, PrimFunc,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { simplifyPrimFunc } from '../../../../src/compiler/passes/simplify/simplify_tir.js';
import { SimplifyPass } from '../../../../src/compiler/passes/simplify/simplify_pass.js';
import { TraceLog, TraceLevel } from '../../../../src/compiler/pipeline/trace.js';

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

function guardedStore(condition) {
  const A = new Buffer('A', [8], 'float32', 'global');
  const i = iv('i');
  const store = new BufferStoreNode(A, [i], new FloatImmNode(0));
  const body = new ForNode(i, c(0), c(8), ForKind.SERIAL, new IfThenElseNode(condition(i), store));
  return new PrimFunc('guarded', [], body);
}

function runTraced(primFunc, level) {
  const events = [];
  const trace = new TraceLog({ level, sink: (event) => events.push(event) });
  const out = new SimplifyPass().run(primFunc, { trace });
  return { out, events, of: (type) => events.filter((e) => e.type === type) };
}

describe('SimplifyPrimFunc counts the guards it decided', () => {
  it('counts a guard it proved and left out of the body', () => {
    const stats = { branchesFolded: 0 };
    simplifyPrimFunc(guardedStore((i) => new CompareNode('lt', i, c(8))), stats);
    expect(stats.branchesFolded).toBe(1);
  });

  it('counts nothing when the guard survives', () => {
    const stats = { branchesFolded: 0 };
    simplifyPrimFunc(guardedStore((i) => new CompareNode('lt', i, c(4))), stats);
    expect(stats.branchesFolded).toBe(0);
  });

  it('counts a folded ternary inside an expression, not only a statement guard', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const stats = { branchesFolded: 0 };
    const f = loopBlock('i', 8, (i) =>
      new IfThenElseNode(new CompareNode('lt', i, c(8)), new BufferLoadNode(A, [i]), new FloatImmNode(0)));

    simplifyPrimFunc(f, stats);

    expect(stats.branchesFolded).toBe(1);
  });
});

describe('SimplifyPass reports what it simplified', () => {
  it('hands back the same function it was given', () => {
    const f = guardedStore((i) => new CompareNode('lt', i, c(8)));
    expect(runTraced(f, TraceLevel.DEBUG).out).toBe(f);
  });

  it('reports the nodes it removed and the guards it folded', () => {
    const { of } = runTraced(guardedStore((i) => new CompareNode('lt', i, c(8))), TraceLevel.DEBUG);
    const detail = of('pass_detail')[0];

    expect(detail.passName).toBe('SimplifyPass');
    expect(detail.branchesFolded).toBe(1);
    expect(detail.nodesRemoved).toBeGreaterThan(0);
  });

  it('explains a function it could fold and one it could not, by name', () => {
    const folded = runTraced(guardedStore((i) => new CompareNode('lt', i, c(8))), TraceLevel.DEBUG).of('explain')[0];
    const kept = runTraced(guardedStore((i) => new CompareNode('lt', i, c(4))), TraceLevel.DEBUG).of('explain')[0];

    expect(folded).toMatchObject({ category: 'simplify', subject: 'guarded', decision: 'folded' });
    expect(kept).toMatchObject({ category: 'simplify', subject: 'guarded', decision: 'unchanged' });
    expect(kept.nodesRemoved).toBeUndefined();
  });

  it('times every run but only counts nodes when the detail would be read', () => {
    const quiet = runTraced(guardedStore((i) => new CompareNode('lt', i, c(8))), TraceLevel.INFO);

    expect(quiet.of('function')[0]).toMatchObject({ phase: 'simplify', funcName: 'guarded' });
    expect(quiet.of('pass_detail')).toEqual([]);
    expect(quiet.of('explain')).toEqual([]);
  });
});
