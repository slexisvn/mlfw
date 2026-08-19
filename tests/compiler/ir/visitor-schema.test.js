import { describe, it, expect } from 'vitest';
import * as tirNodes from '../../../src/compiler/ir/tensor/nodes.js';
import * as lirNodes from '../../../src/compiler/ir/lir/nodes.js';
import { TensorNode } from '../../../src/compiler/ir/tensor/nodes.js';
import { schemaNodeTypes, schemaFor, irChildNodes, childAccessors, walk } from '../../../src/compiler/ir/ir_visitor.js';

function nodeClassNames(...modules) {
  const names = new Set();
  for (const mod of modules) {
    for (const [name, exported] of Object.entries(mod)) {
      if (typeof exported !== 'function') continue;
      if (exported === TensorNode) continue;
      if (!(exported.prototype instanceof TensorNode)) continue;
      names.add(name);
    }
  }
  return names;
}

describe('the traversal schema covers every IR node type', () => {
  it('every TensorNode subclass has a child schema', () => {
    const declared = new Set(schemaNodeTypes());
    const missing = [...nodeClassNames(tirNodes, lirNodes)].filter((name) => !declared.has(name));
    expect(missing, `node types with no ir_visitor schema: ${missing.join(', ')}`).toEqual([]);
  });

  it('the schema declares no type that is not a real node class', () => {
    const real = nodeClassNames(tirNodes, lirNodes);
    const stale = schemaNodeTypes().filter((name) => !real.has(name));
    expect(stale, `schema entries with no matching node class: ${stale.join(', ')}`).toEqual([]);
  });
});

describe('an unknown node type fails the same way on every traversal entry point', () => {
  class UnschemedNode extends TensorNode {}

  it('childAccessors rejects it', () => {
    expect(() => childAccessors(new UnschemedNode())).toThrowError(/no child schema/);
  });

  it('irChildNodes rejects it too, instead of silently reporting no children', () => {
    expect(() => irChildNodes(new UnschemedNode())).toThrowError(/no child schema/);
  });

  it('walk rejects it', () => {
    expect(() => walk(new UnschemedNode(), () => {})).toThrowError(/no child schema/);
  });

  it('non-IR values are still tolerated by irChildNodes', () => {
    expect(irChildNodes(null)).toEqual([]);
    expect(irChildNodes({ type: 'ForNode' })).toEqual([]);
  });
});

describe('schemaFor exposes the field layout the traversal uses', () => {
  it('reports the recursive fields of a loop nest', () => {
    const keys = schemaFor('ForNode').map((f) => f.key);
    expect(keys).toContain('body');
    expect(keys).toContain('extent');
  });

  it('reports nothing for a leaf node', () => {
    expect(schemaFor('IntImmNode')).toEqual([]);
  });

  it('reports undefined for a type it does not know', () => {
    expect(schemaFor('NotANode')).toBeUndefined();
  });
});
