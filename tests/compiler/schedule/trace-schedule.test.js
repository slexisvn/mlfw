import { describe, it, expect } from 'vitest';
import { ScheduleTrace } from '../../../src/compiler/schedule/trace.js';
import { ForNode, BlockNode, BufferStoreNode, VariableNode, IntImmNode, ForKind, PrimFunc } from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';

function buildSplittable() {
  resetVarCounter();
  const buf = new Buffer('A', [16], 'float32', 'global');
  const i = new VariableNode('i', 'int32');
  const store = new BufferStoreNode(buf, [i], new IntImmNode(0));
  const block = new BlockNode('b', [], [], [{ buffer: buf }], store);
  const body = new ForNode(i, new IntImmNode(0), new IntImmNode(16), ForKind.SERIAL, block);
  return new Schedule(new PrimFunc('f', [], body));
}

class FakeSchedule {
  constructor() {
    this._replaying = false;
    this.calls = [];
  }

  ok(...args) {
    this.calls.push(['ok', args]);
  }

  boom() {
    throw new Error('boom');
  }
}

describe('ScheduleTrace.replay', () => {
  it('clears _replaying after each step completes', () => {
    const trace = new ScheduleTrace();
    trace.record('ok', [1]);
    trace.record('ok', [2]);
    const schedule = new FakeSchedule();

    trace.replay(schedule);

    expect(schedule._replaying).toBe(false);
    expect(schedule.calls).toEqual([['ok', [1]], ['ok', [2]]]);
  });

  it('clears _replaying even when a primitive throws', () => {
    const trace = new ScheduleTrace();
    trace.record('boom', []);
    const schedule = new FakeSchedule();

    expect(() => trace.replay(schedule)).toThrow('boom');
    expect(schedule._replaying).toBe(false);
  });

  it('throws on unknown primitive without leaving _replaying set', () => {
    const trace = new ScheduleTrace();
    trace.record('does_not_exist', []);
    const schedule = new FakeSchedule();

    expect(() => trace.replay(schedule)).toThrow(/Unknown schedule primitive/);
    expect(schedule._replaying).toBe(false);
  });

  it('setScope sets a block write-buffer storage scope and is replayable', () => {
    const sch = buildSplittable();
    sch.setScope('b', 'A', 'shared');
    expect(sch.getBlock('b').writes[0].buffer.scope).toBe('shared');
    expect(sch.trace.length).toBe(1);

    const sch2 = buildSplittable();
    expect(sch2.getBlock('b').writes[0].buffer.scope).toBe('global');
    sch.trace.replay(sch2);
    expect(sch2.getBlock('b').writes[0].buffer.scope).toBe('shared');
  });

  it('storageAlign annotates a buffer and is replayable', () => {
    const sch = buildSplittable();
    sch.storageAlign('b', 'A', 0, 8, 0);
    expect(sch.getBlock('b').writes[0].buffer.storageAlign).toEqual({ axis: 0, factor: 8, offset: 0 });

    const sch2 = buildSplittable();
    sch.trace.replay(sch2);
    expect(sch2.getBlock('b').writes[0].buffer.storageAlign).toEqual({ axis: 0, factor: 8, offset: 0 });
  });

  it('replays a recorded split onto a fresh schedule by resolving loop names', () => {
    const sch1 = buildSplittable();
    sch1.split(sch1.getLoops('b')[0], 4);
    const after1 = sch1.state.allLoopVarNames();
    expect(after1).not.toContain('i');
    expect(sch1.trace.length).toBe(1);

    const sch2 = buildSplittable();
    expect(sch2.state.allLoopVarNames()).toEqual(['i']);
    sch1.trace.replay(sch2);

    const after2 = sch2.state.allLoopVarNames();
    expect(after2).not.toContain('i');
    expect(after2.length).toBe(after1.length);
  });
});
