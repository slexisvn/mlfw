import { describe, it, expect } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode, IfThenElseNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, CompareNode, ForKind, PrimFunc, SeqNode,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import { simplifyPrimFunc } from '../../../../src/compiler/passes/simplify/simplify_tir.js';
import { SimplifyPass } from '../../../../src/compiler/passes/simplify/simplify_pass.js';
import { TraceLog, TraceLevel } from '../../../../src/compiler/support/trace.js';

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

describe('SimplifyPrimFunc: what the modular and fact layers add', () => {
  it('folds ((i % 3) * 8) % 8 to 0, which the affine split cannot see through', () => {
    const A = new Buffer('A', [64], 'float32', 'global');
    const i = iv('i');
    const wrapped = new MathOpNode('*', new MathOpNode('%', i, c(3)), c(8));
    const store = new BufferStoreNode(A, [new MathOpNode('%', wrapped, c(8))], new FloatImmNode(0));
    const f = new PrimFunc('f', [], new ForNode(i, c(0), c(8), ForKind.SERIAL, store));
    simplifyPrimFunc(f);
    expect(f.body.body.indices[0]).toMatchObject({ type: 'IntImmNode', value: 0 });
  });

  it('folds ((i % 3) * 4 + j * 8) // 4 into the exact quotient', () => {
    const A = new Buffer('A', [64], 'float32', 'global');
    const i = iv('i');
    const j = iv('j');
    const wrapped = new MathOpNode('*', new MathOpNode('%', i, c(3)), c(4));
    const scaled = new MathOpNode('+', wrapped, new MathOpNode('*', j, c(8)));
    const store = new BufferStoreNode(A, [new MathOpNode('//', scaled, c(4))], new FloatImmNode(0));
    const inner = new ForNode(j, c(0), c(4), ForKind.SERIAL, store);
    const f = new PrimFunc('f', [], new ForNode(i, c(0), c(4), ForKind.SERIAL, inner));
    simplifyPrimFunc(f);
    const index = f.body.body.body.indices[0];
    expect(countNodes(index, 'MathOpNode')).toBe(3);
    expect(index).toMatchObject({ op: '+' });
  });

  it('keeps a division through a non-affine term it cannot prove exact', () => {
    const A = new Buffer('A', [64], 'float32', 'global');
    const i = iv('i');
    const wrapped = new MathOpNode('*', new MathOpNode('%', i, c(3)), c(8));
    const index = new MathOpNode('//', new MathOpNode('+', wrapped, c(3)), c(4));
    const store = new BufferStoreNode(A, [index], new FloatImmNode(0));
    const f = new PrimFunc('f', [], new ForNode(i, c(0), c(8), ForKind.SERIAL, store));
    simplifyPrimFunc(f);
    expect(countNodes(f.body.body.indices[0], 'MathOpNode')).toBeGreaterThan(1);
  });

  it('uses the taken branch condition to fold a guard inside it', () => {
    const A = new Buffer('A', [16], 'float32', 'global');
    const i = iv('i');
    const inner = new IfThenElseNode(new CompareNode('lt', i, c(8)), new BufferStoreNode(A, [i], new FloatImmNode(1)));
    const outer = new IfThenElseNode(new CompareNode('lt', i, c(4)), inner);
    const f = new PrimFunc('f', [], new ForNode(i, c(0), c(16), ForKind.SERIAL, outer));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(1);
    expect(f.body.body.condition).toMatchObject({ direction: 'lt' });
    expect(f.body.body.condition.b).toMatchObject({ value: 4 });
  });

  it('uses the untaken branch condition inside the else arm', () => {
    const A = new Buffer('A', [16], 'float32', 'global');
    const i = iv('i');
    const elseArm = new IfThenElseNode(new CompareNode('ge', i, c(4)), new BufferStoreNode(A, [i], new FloatImmNode(2)));
    const outer = new IfThenElseNode(
      new CompareNode('lt', i, c(4)),
      new BufferStoreNode(A, [i], new FloatImmNode(1)),
      elseArm,
    );
    const f = new PrimFunc('f', [], new ForNode(i, c(0), c(16), ForKind.SERIAL, outer));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(1);
    expect(countNodes(f.body, 'BufferStoreNode')).toBe(2);
  });

  it('drops the assumption again once the branch ends', () => {
    const A = new Buffer('A', [16], 'float32', 'global');
    const i = iv('i');
    const taken = new IfThenElseNode(new CompareNode('lt', i, c(4)), new BufferStoreNode(A, [i], new FloatImmNode(1)));
    const after = new IfThenElseNode(new CompareNode('lt', i, c(8)), new BufferStoreNode(A, [i], new FloatImmNode(2)));
    const f = new PrimFunc('f', [], new ForNode(i, c(0), c(16), ForKind.SERIAL, new SeqNode([taken, after])));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(2);
  });

  it('folds a guard that only holds because of a symbolic loop extent', () => {
    const A = new Buffer('A', [16], 'float32', 'global');
    const i = iv('i');
    const n = iv('n');
    const guarded = new IfThenElseNode(new CompareNode('lt', i, n), new BufferStoreNode(A, [i], new FloatImmNode(0)));
    const f = new PrimFunc('f', [], new ForNode(i, c(0), n, ForKind.SERIAL, guarded));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(0);
    expect(countNodes(f.body, 'BufferStoreNode')).toBe(1);
  });

  it('keeps a guard a symbolic extent does not justify', () => {
    const A = new Buffer('A', [16], 'float32', 'global');
    const i = iv('i');
    const n = iv('n');
    const guarded = new IfThenElseNode(new CompareNode('lt', plusOne(i), n), new BufferStoreNode(A, [i], new FloatImmNode(0)));
    const f = new PrimFunc('f', [], new ForNode(i, c(0), n, ForKind.SERIAL, guarded));
    simplifyPrimFunc(f);
    expect(countNodes(f.body, 'IfThenElseNode')).toBe(1);
  });
});

function plusOne(i) {
  return new MathOpNode('+', i, c(1));
}
