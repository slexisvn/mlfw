import { describe, it, expect } from 'vitest';
import { ScheduleTrace } from '../../../src/compiler/schedule/trace.js';
import { ForNode, BlockNode, BufferStoreNode, VariableNode, IntImmNode, ForKind, PrimFunc } from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { SchedulePolicy } from '../../../src/compiler/schedule/rules.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { TraceLevel } from '../../../src/compiler/pipeline/trace.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { clonePrimFunc } from '../../../src/compiler/autotune/tune_ir.js';
import { printTensorIR } from '../../../src/compiler/ir/tensor/printer.js';

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

const mmT = new TensorType([64, 64], ScalarType.F32);

function matmulGraph() {
  return buildFunction('mm', [mmT, mmT], [mmT], (b, args) => {
    const d = b.dot(args[0], args[1], [1], [0], [], []).getResult(0);
    b.returnOp([b.tanh(d).getResult(0)]);
  });
}

function loweredPrimFunc() {
  return lowerGraphToPrimFunc(matmulGraph(), CPUTarget());
}

function alphaNormalized(primFunc) {
  const names = new Map();
  return printTensorIR(primFunc).replace(/\b[A-Za-z_][A-Za-z0-9_]*_\d+\b/g, (name) => {
    if (!names.has(name)) names.set(name, `v${names.size}`);
    return names.get(name);
  });
}

function countLoops(primFunc) {
  return printTensorIR(primFunc).split('\n').filter((line) => line.includes('for ')).length;
}

function firstBlockName(schedule) {
  const match = printTensorIR(schedule.func).match(/block\s+([A-Za-z0-9_]+)/);
  expect(match).not.toBeNull();
  return match[1];
}

function splitFirstLoop(schedule) {
  const blockName = firstBlockName(schedule);
  schedule.split(schedule.getLoops(blockName)[0], 8);
  return blockName;
}

describe('replaying a schedule trace one primitive at a time', () => {
  it('calls the observer once per step, in the order the steps were recorded', () => {
    const schedule = new Schedule(loweredPrimFunc());
    splitFirstLoop(schedule);

    const replayed = new Schedule(clonePrimFunc(loweredPrimFunc()));
    const seen = [];
    schedule.trace.replayEach(replayed, (step, index) => seen.push([index, step.primitive]));

    expect(seen).toEqual([[0, 'split']]);
  });

  it('applies each step before handing it to the observer', () => {
    const schedule = new Schedule(loweredPrimFunc());
    splitFirstLoop(schedule);

    const replayed = new Schedule(clonePrimFunc(loweredPrimFunc()));
    const before = countLoops(replayed.func);
    let atObserver = null;
    schedule.trace.replayEach(replayed, () => { atObserver = countLoops(replayed.func); });

    expect(atObserver).toBe(before + 1);
  });

  it('lands on the same loop nest the recording schedule produced', () => {
    const schedule = new Schedule(loweredPrimFunc());
    const blockName = splitFirstLoop(schedule);
    schedule.vectorize(schedule.getLoops(blockName).at(-1));

    const replayed = new Schedule(clonePrimFunc(loweredPrimFunc()));
    schedule.trace.replay(replayed);

    expect(alphaNormalized(replayed.func)).toBe(alphaNormalized(schedule.func));
  });

  it('reaches loops an earlier primitive created, whose names the replay mints afresh', () => {
    const schedule = new Schedule(loweredPrimFunc());
    new SchedulePolicy(CPUTarget(), null, null).applyToAllBlocks(schedule);
    const steps = schedule.trace.serialize();
    const created = new Set(steps.flatMap((step) => step.produced));

    expect(created.size).toBeGreaterThan(0);
    expect(steps.some((step) => step.args.some((arg) => created.has(arg)))).toBe(true);

    const replayed = new Schedule(loweredPrimFunc());
    schedule.trace.replay(replayed);

    expect(alphaNormalized(replayed.func)).toBe(alphaNormalized(schedule.func));
  });

  it('survives a round trip through the serialized form', () => {
    const schedule = new Schedule(loweredPrimFunc());
    splitFirstLoop(schedule);

    const restored = ScheduleTrace.deserialize(JSON.parse(JSON.stringify(schedule.trace.serialize())));
    const replayed = new Schedule(clonePrimFunc(loweredPrimFunc()));
    restored.replay(replayed);

    expect(alphaNormalized(replayed.func)).toBe(alphaNormalized(schedule.func));
  });
});

describe('the scheduling pass publishing what it applied', () => {
  function published(enabled) {
    const events = [];
    compileGraph(matmulGraph(), CPUTarget(), {
      scheduling: { enabled },
      trace: { level: TraceLevel.DEBUG, sink: (e) => events.push(e) },
    });
    return events.filter((e) => e.type === 'schedule_trace');
  }

  it('publishes a step sequence that replays into the nest the pass scheduled', () => {
    const events = published(true);
    expect(events).toHaveLength(1);

    const replayed = new Schedule(loweredPrimFunc());
    ScheduleTrace.deserialize(events[0].steps).replay(replayed);

    const applied = new Schedule(loweredPrimFunc());
    new SchedulePolicy(CPUTarget(), null, null).applyToAllBlocks(applied);

    expect(alphaNormalized(replayed.func)).toBe(alphaNormalized(applied.func));
  });

  it('says nothing when scheduling is turned off', () => {
    expect(published(false)).toHaveLength(0);
  });
});
