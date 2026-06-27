import { describe, it, expect } from 'vitest';
import {
  TensorNode, ForNode, SeqNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, VariableNode, IntImmNode, IfThenElseNode, CompareNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { LIRFlatStoreNode, LIRFunc } from '../../../src/compiler/ir/lir/nodes.js';
import { walk, transform, collect, some, find, childAccessors, isIRNode } from '../../../src/compiler/ir/ir_visitor.js';

const buf = (name) => ({ name, dtype: 'f32' });

function sampleTree() {
  const i = new VariableNode('i', 'i32');
  const n = new VariableNode('n', 'i32');
  const load = new BufferLoadNode(buf('B'), [new VariableNode('i', 'i32')]);
  const add = new MathOpNode('+', load, new IntImmNode(2));
  const store = new BufferStoreNode(buf('A'), [new VariableNode('i', 'i32')], add);
  const body = new SeqNode([store]);
  const loop = new ForNode(i, new IntImmNode(0), n, 'serial', body);
  return { loop, body, store, add, load, n, i };
}

describe('ir_visitor — schema-driven traversal', () => {
  it('descends UNREGISTERED expr children (ForNode.min/extent, BufferStore.indices/value, MathOp.a/b)', () => {
    const { loop, n } = sampleTree();
    const types = collect(loop, () => true).map((x) => x.type);
    expect(types).toContain('IntImmNode');
    expect(types).toContain('BufferLoadNode');
    expect(types).toContain('MathOpNode');
    expect(types).toContain('BufferStoreNode');
    expect(collect(loop, (x) => x === n).length).toBe(1);
    expect(collect(loop, (x) => x.type === 'IntImmNode' && x.value === 0).length).toBe(1);
  });

  it('handles nullable MathOp.b (unary op) without descending into null', () => {
    const unary = new MathOpNode('neg', new IntImmNode(5), null);
    const nodes = collect(unary, () => true);
    expect(nodes.map((x) => x.type)).toEqual(['MathOpNode', 'IntImmNode']);
  });

  it('kinds:"stmt" stops at statement children, kinds:"expr" stops at expressions', () => {
    const { loop } = sampleTree();
    const stmtOnly = collect(loop, () => true, { kinds: 'stmt' }).map((x) => x.type);
    expect(stmtOnly).toContain('SeqNode');
    expect(stmtOnly).toContain('BufferStoreNode');
    expect(stmtOnly).not.toContain('MathOpNode');
    expect(stmtOnly).not.toContain('IntImmNode');
  });

  it('some/find early-exit on first match', () => {
    const { loop, store } = sampleTree();
    expect(some(loop, (x) => x.type === 'BufferStoreNode')).toBe(true);
    expect(some(loop, (x) => x.type === 'WhileNode')).toBe(false);
    expect(find(loop, (x) => x.type === 'BufferStoreNode')).toBe(store);
    expect(find(loop, (x) => x.type === 'CastNode')).toBe(null);
  });

  it('pre returning false prunes a subtree', () => {
    const { loop } = sampleTree();
    const seen = [];
    walk(loop, (n) => { seen.push(n.type); if (n.type === 'SeqNode') return false; });
    expect(seen).toContain('ForNode');
    expect(seen).toContain('SeqNode');
    expect(seen).not.toContain('BufferStoreNode');
  });

  it('descendParams is opt-in (default skips PrimFunc params / leaf bind sites not double-counted)', () => {
    const { loop } = sampleTree();
    const withBind = collect(loop, (x) => x.type === 'VariableNode').length;
    const noBind = collect(loop, (x) => x.type === 'VariableNode', { bindVars: false }).length;
    expect(withBind).toBeGreaterThan(noBind);
  });

  it('works over LIR nodes (shared expr layer)', () => {
    const store = new LIRFlatStoreNode(buf('A'), new VariableNode('off', 'i32'),
      new MathOpNode('*', new IntImmNode(3), new IntImmNode(4)), 'f32');
    const fn = new LIRFunc('f', [], store, new Map(), [], new Map(), null);
    const imms = collect(fn, (x) => x.type === 'IntImmNode').map((x) => x.value).sort();
    expect(imms).toEqual([3, 4]);
  });

  it('throws on a node type with no child schema (catches omissions loudly)', () => {
    class MysteryNode extends TensorNode {}
    expect(() => walk(new MysteryNode(), () => {})).toThrow(/no child schema/);
  });
});

describe('ir_visitor — transform', () => {
  it('replaces a node and rebuilds parent links so replaceWith still works', () => {
    const { loop, add } = sampleTree();
    transform(loop, (node) => {
      if (node.type === 'IntImmNode' && node.value === 2) return new IntImmNode(99);
      return undefined;
    });
    expect(add.b.value).toBe(99);
    expect(add.b._parent).toBe(add);
    expect(add.b._parentKey).toBe('b');
    const ok = add.b.replaceWith(new IntImmNode(7));
    expect(ok).toBe(true);
    expect(add.b.value).toBe(7);
  });

  it('is share-on-no-change: returns the same node when nothing matched', () => {
    const { loop } = sampleTree();
    const out = transform(loop, () => undefined);
    expect(out).toBe(loop);
  });

  it('rewrites array children in place (BufferStore.indices)', () => {
    const { loop, store } = sampleTree();
    transform(loop, (node) => {
      if (node.type === 'VariableNode' && node.name === 'i') return new VariableNode('j', 'i32');
      return undefined;
    }, { bindVars: false });
    expect(store.indices[0].name).toBe('j');
  });
});
