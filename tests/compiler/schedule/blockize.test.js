import { describe, it, expect } from 'vitest';
import { ForNode, BlockNode, BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, ForKind, PrimFunc } from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';

function build2D() {
  resetVarCounter();
  const A = new Buffer('A', [4, 4], 'float32', 'global');
  const C = new Buffer('C', [4, 4], 'float32', 'global');
  const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
  const store = new BufferStoreNode(C, [i, j], new BufferLoadNode(A, [i, j]));
  const block = new BlockNode('b', [], [{ buffer: A }], [{ buffer: C }], store);
  const inner = new ForNode(j, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
  const outer = new ForNode(i, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, inner);
  return new Schedule(new PrimFunc('f', [], outer));
}

describe('Schedule.blockize', () => {
  it('wraps a loop-nest into a composite block with aggregated reads/writes', () => {
    const sch = build2D();
    const wrapper = sch.blockize('i');

    expect(wrapper.type).toBe('BlockNode');
    expect(wrapper.name).toBe('blockized_i');
    expect(wrapper.body.type).toBe('ForNode');
    expect(wrapper.body.loopVar.name).toBe('i');
    expect(wrapper.reads.map(r => r.buffer.name)).toContain('A');
    expect(wrapper.writes.map(w => w.buffer.name)).toContain('C');
    expect(sch.func.body).toBe(wrapper);
    expect(sch.verify()).toEqual([]);
  });

  it('is replayable by loop name onto a fresh schedule', () => {
    const sch = build2D();
    sch.blockize('i');
    const trace = sch.getTrace();

    const fresh = build2D();
    trace.replay(fresh);

    expect(fresh.func.body.type).toBe('BlockNode');
    expect(fresh.func.body.name).toBe('blockized_i');
    expect(fresh.func.body.body.loopVar.name).toBe('i');
  });

  it('rejects a non-loop target', () => {
    const sch = build2D();
    expect(() => sch.blockize('not_a_loop_var')).toThrow(/blockize expects a loop/);
  });
});
