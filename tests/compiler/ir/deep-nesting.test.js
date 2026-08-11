import { describe, it, expect } from 'vitest';
import {
  AllocateNode, SeqNode, BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, MathOpNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { walk, transform, collect } from '../../../src/compiler/ir/ir_visitor.js';

const DEPTH = 20000;

function deepAllocateChain(depth) {
  const out = new Buffer('out', [1], 'float32', 'global');
  let body = new BufferStoreNode(out, [new IntImmNode(0)], new IntImmNode(0));
  for (let d = depth - 1; d >= 0; d--) {
    const buf = new Buffer(`t${d}`, [1], 'float32', 'local');
    body = new AllocateNode(buf, 'local', new SeqNode([body]));
  }
  return body;
}

describe('IR traversal survives nesting deeper than the JS call stack', () => {
  it('walk visits every node of a deep allocate chain in pre/post order', () => {
    const root = deepAllocateChain(DEPTH);
    let allocates = 0;
    let maxDepth = 0;
    let pres = 0;
    let posts = 0;
    walk(root, {
      pre(node, ctx) {
        pres++;
        if (node.type === 'AllocateNode') allocates++;
        if (ctx.depth > maxDepth) maxDepth = ctx.depth;
      },
      post() { posts++; },
    });
    expect(allocates).toBe(DEPTH);
    expect(maxDepth).toBeGreaterThanOrEqual(DEPTH * 2);
    expect(posts).toBe(pres);
  });

  it('collect reaches the innermost statement of a deep chain', () => {
    const stores = collect(deepAllocateChain(DEPTH), (n) => n.type === 'BufferStoreNode');
    expect(stores).toHaveLength(1);
  });

  it('transform rewrites the innermost node of a deep chain and keeps the chain intact', () => {
    const root = deepAllocateChain(DEPTH);
    const rewritten = transform(root, (n) => (
      n.type === 'IntImmNode' && n.value === 0 ? new IntImmNode(7) : n
    ));
    let node = rewritten;
    let seen = 0;
    while (node && node.type === 'AllocateNode') { seen++; node = node.body.stmts[0]; }
    expect(seen).toBe(DEPTH);
    expect(node.type).toBe('BufferStoreNode');
    expect(node.value.value).toBe(7);
    expect(node.indices[0].value).toBe(7);
  });

  it('transform replaces the root itself when the visitor returns a new node', () => {
    const A = new Buffer('A', [4], 'float32', 'global');
    const root = new MathOpNode('+', new BufferLoadNode(A, [new IntImmNode(1)]), new IntImmNode(2));
    const replaced = transform(root, (n) => (n.type === 'MathOpNode' ? new IntImmNode(99) : n));
    expect(replaced.type).toBe('IntImmNode');
    expect(replaced.value).toBe(99);
  });
});
