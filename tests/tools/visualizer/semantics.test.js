import { describe, it, expect } from 'vitest';
import {
  PrimFunc, ForNode, SeqNode, BufferStoreNode, BufferLoadNode,
  MathOpNode, VariableNode, IntImmNode, EvaluateNode, ForKind,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { semanticReport } from '../../../tools/visualizer/src/worker/semantics.js';

const N = 4;

function buffers() {
  return {
    out: new Buffer('out', [N, N], 'float32', 'global'),
    lhs: new Buffer('lhs', [N, N], 'float32', 'global'),
    rhs: new Buffer('rhs', [N, N], 'float32', 'global'),
  };
}

function elementwise({ op = '+', shift = 0, swap = false } = {}) {
  const { out, lhs, rhs } = buffers();

  const inner = (i, j) => new BufferStoreNode(
    out,
    [i, new MathOpNode('+', j, new IntImmNode(shift))],
    new MathOpNode(op, new BufferLoadNode(lhs, [i, j]), new BufferLoadNode(rhs, [i, j])),
  );

  const i = new VariableNode('i', 'int32');
  const j = new VariableNode('j', 'int32');
  const body = inner(i, j);

  const jLoop = new ForNode(j, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, body);
  const iLoop = new ForNode(i, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, jLoop);

  const outerFirst = swap
    ? new ForNode(j, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL,
      new ForNode(i, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, body))
    : iLoop;

  return new PrimFunc('elementwise', [], outerFirst);
}

function report(before, after) {
  return semanticReport([before], [after]);
}

describe('semantic diff of a loop nest', () => {
  it('calls an identical nest preserved', () => {
    const found = report(elementwise(), elementwise());

    expect(found.ran).toBe(true);
    expect(found.changedCount).toBe(0);
    expect(found.droppedCount).toBe(0);
    expect(found.addedCount).toBe(0);
    expect(found.reordered).toBe(false);
    expect(found.compared).toBe(N * N);
    expect(found.verdict).toContain("preserved the program's meaning");
  });

  it('sees a loop interchange as a reorder, not a meaning change', () => {
    const found = report(elementwise(), elementwise({ swap: true }));

    expect(found.changedCount).toBe(0);
    expect(found.droppedCount).toBe(0);
    expect(found.reordered).toBe(true);
    expect(found.verdict).toContain('different order');
  });

  it('catches an operator that changed and names the first cell', () => {
    const found = report(elementwise(), elementwise({ op: '-' }));

    expect(found.changedCount).toBe(N * N);
    expect(found.changed[0].cell).toBe('out[0, 0]');
    expect(found.changed[0].before).not.toBe(found.changed[0].after);
    expect(found.verdict).toContain('changed what the program computes');
  });

  it('catches an off-by-one write index in both the values and the cell set', () => {
    const found = report(elementwise(), elementwise({ shift: 1 }));

    expect(found.droppedCount).toBeGreaterThan(0);
    expect(found.addedCount).toBeGreaterThan(0);
    expect(found.dropped[0]).toBe('out[0, 0]');
    expect(found.changedCount).toBeGreaterThan(0);
    expect(found.verdict).toContain('changed what the program computes');
  });

  it('reports a pure cell-set change when the values that survive still agree', () => {
    const out = new Buffer('out', [2], 'float32', 'global');
    const one = new PrimFunc('one', [], new BufferStoreNode(out, [new IntImmNode(0)], new IntImmNode(5)));
    const two = new PrimFunc('two', [], new SeqNode([
      new BufferStoreNode(out, [new IntImmNode(0)], new IntImmNode(5)),
      new BufferStoreNode(out, [new IntImmNode(1)], new IntImmNode(9)),
    ]));

    const found = report(one, two);
    expect(found.changedCount).toBe(0);
    expect(found.addedCount).toBe(1);
    expect(found.verdict).toContain('the set of cells changed');
  });

  it('reads a load back as the value most recently stored there', () => {
    const out = new Buffer('out', [1], 'float32', 'global');
    const seven = new PrimFunc('chain', [], new SeqNode([
      new BufferStoreNode(out, [new IntImmNode(0)], new IntImmNode(7)),
    ]));
    const doubled = new PrimFunc('chain', [], new SeqNode([
      new BufferStoreNode(out, [new IntImmNode(0)], new IntImmNode(7)),
      new BufferStoreNode(out, [new IntImmNode(0)],
        new MathOpNode('*', new BufferLoadNode(out, [new IntImmNode(0)]), new IntImmNode(2))),
    ]));

    const found = report(seven, doubled);
    expect(found.changed).toEqual([{ cell: 'out[0]', before: 7, after: 14 }]);
  });

  it('says why it could not run instead of guessing', () => {
    const out = new Buffer('out', [N], 'float32', 'global');
    const symbolic = new VariableNode('n', 'int32');
    const i = new VariableNode('i', 'int32');
    const dynamic = new PrimFunc('dynamic', [], new ForNode(
      i, new IntImmNode(0), symbolic, ForKind.SERIAL,
      new BufferStoreNode(out, [i], new IntImmNode(1)),
    ));

    const found = report(dynamic, dynamic);
    expect(found.ran).toBe(false);
    expect(found.reason).toContain('shape only known at run time');
    expect(found.verdict).toBe(found.reason);
  });

  it('refuses to cry miscompile when a buffer was folded into another', () => {
    const kept = new Buffer('kept', [2], 'float32', 'global');
    const temp = new Buffer('temp', [2], 'float32', 'global');

    const before = new PrimFunc('plan', [], new SeqNode([
      new BufferStoreNode(kept, [new IntImmNode(0)], new IntImmNode(1)),
      new BufferStoreNode(temp, [new IntImmNode(0)], new IntImmNode(2)),
    ]));
    const after = new PrimFunc('plan', [], new SeqNode([
      new BufferStoreNode(kept, [new IntImmNode(0)], new IntImmNode(1)),
      new BufferStoreNode(kept, [new IntImmNode(0)], new IntImmNode(2)),
    ]));

    const found = report(before, after);
    expect(found.storageReused).toBe(true);
    expect(found.vanishedBuffers).toEqual(['temp']);
    expect(found.changedCount).toBe(1);
    expect(found.verdict).toContain('reuses storage');
    expect(found.verdict).not.toContain('changed what the program computes');
  });

  it('still calls a plain new temporary a new temporary', () => {
    const kept = new Buffer('kept', [1], 'float32', 'global');
    const fresh = new Buffer('fresh', [1], 'float32', 'global');
    const before = new PrimFunc('t', [], new BufferStoreNode(kept, [new IntImmNode(0)], new IntImmNode(1)));
    const after = new PrimFunc('t', [], new SeqNode([
      new BufferStoreNode(kept, [new IntImmNode(0)], new IntImmNode(1)),
      new BufferStoreNode(fresh, [new IntImmNode(0)], new IntImmNode(9)),
    ]));

    const found = report(before, after);
    expect(found.storageReused).toBe(false);
    expect(found.newBuffers).toEqual(['fresh']);
    expect(found.changedCount).toBe(0);
    expect(found.verdict).toContain('1 buffer is new');
  });

  it('refuses to decide when it ran out of budget', () => {
    const out = new Buffer('out', [1], 'float32', 'global');
    const i = new VariableNode('i', 'int32');
    const j = new VariableNode('j', 'int32');
    const huge = (value) => new PrimFunc('huge', [], new ForNode(
      i, new IntImmNode(0), new IntImmNode(4096), ForKind.SERIAL,
      new ForNode(j, new IntImmNode(0), new IntImmNode(4096), ForKind.SERIAL,
        new BufferStoreNode(out, [new IntImmNode(0)], new IntImmNode(value))),
    ));

    const found = report(huge(1), huge(2));
    expect(found.truncated).toBe(true);
    expect(found.verdict).toContain('Inconclusive');
    expect(found.verdict).not.toContain('changed what the program computes');
  });

  it('names the node it has no rule for', () => {
    const odd = new PrimFunc('odd', [], new EvaluateNode(new VariableNode('x', 'int32')));
    const broken = { ...odd, body: { type: 'MysteryNode' } };

    expect(report(odd, broken).reason).toContain("MysteryNode");
  });
});
