import { describe, it, expect } from 'vitest';
import { MemoryPlanner } from '../../../../src/compiler/passes/memory/memory_planning.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, ForNode, ForKind,
  IntImmNode, MathOpNode, VariableNode, BlockRealizeNode, AllocateNode
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { compileGraph } from '../../../../src/compiler/pipeline/compiler.js';
import { TraceLevel } from '../../../../src/compiler/support/trace.js';

function makeVar(name) {
  return new VariableNode(name, 'int32');
}

function makeBind(name) {
  return new BlockRealizeNode(makeVar(name + '_v'), makeVar(name));
}

function makeCoveringBody(writeBuf, readBufs) {
  const iv = makeVar(`${writeBuf.name}_i`);
  const value = readBufs.length > 0
    ? new BufferLoadNode(readBufs[0], [iv])
    : new IntImmNode(0);
  const store = new BufferStoreNode(writeBuf, [iv], value);
  return new ForNode(iv, new IntImmNode(0), new IntImmNode(writeBuf.shape[0]), ForKind.SERIAL, store);
}

function makeBlock(name, reads, writes) {
  const binds = [makeBind('i')];
  const body = writes.length > 0
    ? makeCoveringBody(writes[0].buffer, reads.map((r) => r.buffer))
    : new IntImmNode(0);
  return new BlockNode(name, binds, reads, writes, body);
}

function makePrimFunc(body, paramBuffers = []) {
  const params = paramBuffers.map((_, i) => makeVar(`arg_${i}`));
  const bufferMap = new Map();
  for (let i = 0; i < paramBuffers.length; i++) {
    bufferMap.set(params[i], paramBuffers[i]);
  }
  return new PrimFunc('test', params, body, bufferMap);
}

describe('MemoryPlanner.plan', () => {
  it('returns a MemoryPlan with assignment and liveness', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const tempBuf = new Buffer('temp', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: tempBuf }]);
    const block1 = makeBlock('b1', [{ buffer: tempBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const plan = planner.plan(pf);

    expect(plan.assignment).toBeDefined();
    expect(plan.liveness).toBeDefined();
    expect(plan.peakMemory()).toBeGreaterThan(0);
  });

  it('peak memory equals single buffer when temporaries are sequential', () => {
    const paramBuf = new Buffer('param', [16], 'f32', 'global');
    const t1 = new Buffer('t1', [16], 'f32', 'global');
    const t2 = new Buffer('t2', [16], 'f32', 'global');
    const outBuf = new Buffer('out', [16], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: t1 }]);
    const block1 = makeBlock('b1', [{ buffer: t1 }], [{ buffer: t2 }]);
    const block2 = makeBlock('b2', [{ buffer: t2 }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const plan = planner.plan(pf);

    const singleBufBytes = 16 * 4;
    const aligned = Math.ceil(singleBufBytes / 64) * 64;
    expect(plan.peakMemory()).toBe(aligned);
  });

  it('peak memory doubles when temporaries overlap', () => {
    const paramBuf = new Buffer('param', [16], 'f32', 'global');
    const t1 = new Buffer('t1', [16], 'f32', 'global');
    const t2 = new Buffer('t2', [16], 'f32', 'global');
    const outBuf = new Buffer('out', [16], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: t1 }]);
    const block1 = makeBlock('b1', [{ buffer: paramBuf }], [{ buffer: t2 }]);
    const binds = [makeBind('i')];
    const body = new BufferStoreNode(outBuf, [], new MathOpNode('+', new BufferLoadNode(t1, []), new BufferLoadNode(t2, [])));
    const block2 = new BlockNode('b2', binds, [{ buffer: t1 }, { buffer: t2 }], [{ buffer: outBuf }], body);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const plan = planner.plan(pf);

    const singleBufBytes = 16 * 4;
    const aligned = Math.ceil(singleBufBytes / 64) * 64;
    expect(plan.peakMemory()).toBe(aligned * 2);
  });

  it('custom alignment is respected', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const tempBuf = new Buffer('temp', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: tempBuf }]);
    const block1 = makeBlock('b1', [{ buffer: tempBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner({ alignment: 256 });
    const plan = planner.plan(pf);

    expect(plan.peakMemory() % 256).toBe(0);
  });
});

describe('MemoryPlanner with inplace disabled', () => {
  it('does not produce inplace candidates when enableInplace is false', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }]);
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner({ enableInplace: false });
    const plan = planner.plan(pf);

    expect(plan.inplaceCandidates.length).toBe(0);
  });
});

describe('MemoryPlanner.planAndRewrite', () => {
  it('inserts AllocateNode wrappers for temporary buffers', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const tempBuf = new Buffer('temp', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: tempBuf }]);
    const block1 = makeBlock('b1', [{ buffer: tempBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const { func, plan } = planner.planAndRewrite(pf);

    expect(func.body).toBeInstanceOf(AllocateNode);
    expect(func.body.buffer).toBe(tempBuf);
  });

  it('does not insert AllocateNode when there are no temporaries', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');
    const block = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: outBuf }]);
    const pf = makePrimFunc(block, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const { func } = planner.planAndRewrite(pf);

    expect(func.body).toBeInstanceOf(BlockNode);
  });

  it('does not insert AllocateNode for inplace buffers', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const srcBuf = new Buffer('src', [8], 'f32', 'global');
    const dstBuf = new Buffer('dst', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: srcBuf }]);
    const block1 = makeBlock('b1', [{ buffer: srcBuf }], [{ buffer: dstBuf }]);
    const block2 = makeBlock('b2', [{ buffer: dstBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const { func, plan } = planner.planAndRewrite(pf);

    let allocCount = 0;
    let cur = func.body;
    while (cur instanceof AllocateNode) {
      allocCount++;
      cur = cur.body;
    }

    const inplaceCount = plan.inplaceCandidates.length;
    const totalTemps = plan.liveness.getTemporaries().length;
    expect(allocCount).toBeLessThanOrEqual(totalTemps - inplaceCount);
  });
});

describe('MemoryPlan.getReport', () => {
  it('report contains correct counts', () => {
    const paramBuf = new Buffer('param', [8], 'f32', 'global');
    const tempBuf = new Buffer('temp', [8], 'f32', 'global');
    const outBuf = new Buffer('out', [8], 'f32', 'global');

    const block0 = makeBlock('b0', [{ buffer: paramBuf }], [{ buffer: tempBuf }]);
    const block1 = makeBlock('b1', [{ buffer: tempBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const planner = new MemoryPlanner();
    const plan = planner.plan(pf);
    const report = plan.getReport();

    expect(report.peakMemory).toBeGreaterThan(0);
    expect(report.totalTemporaries).toBe(1);
    expect(report.assignments.size).toBeGreaterThanOrEqual(1);
  });
});

describe('buffer lifetimes reported by the memory plan', () => {
  const vecT = new TensorType([64], ScalarType.F32);

  function planOf(func) {
    const events = [];
    compileGraph(func, CPUTarget(), {
      fusion: { enabled: false },
      trace: { level: TraceLevel.DEBUG, sink: (e) => events.push(e) },
    });
    const plans = events.filter((e) => e.type === 'memory_plan' && e.funcName === func.name);
    expect(plans).toHaveLength(1);
    return plans[0];
  }

  function chainOfFour() {
    return buildFunction('chain', [vecT, vecT], [vecT], (b, args) => {
      const [x, y] = args;
      const a = b.mul(x, y).getResult(0);
      const c = b.add(a, x).getResult(0);
      const d = b.mul(c, c).getResult(0);
      b.returnOp([b.tanh(d).getResult(0)]);
    });
  }

  function twoLiveAtOnce() {
    return buildFunction('diamond', [vecT, vecT], [vecT], (b, args) => {
      const [x, y] = args;
      const left = b.mul(x, y).getResult(0);
      const right = b.add(x, y).getResult(0);
      b.returnOp([b.mul(left, right).getResult(0)]);
    });
  }

  it('reports one interval per temporary, ordered by first use', () => {
    const plan = planOf(chainOfFour());

    expect(plan.buffers).toHaveLength(3);
    expect(new Set(plan.buffers.map((b) => b.name)).size).toBe(plan.buffers.length);
    for (let i = 1; i < plan.buffers.length; i++) {
      expect(plan.buffers[i].firstUse).toBeGreaterThanOrEqual(plan.buffers[i - 1].firstUse);
    }
  });

  it('keeps every interval inside the statement order it was measured against', () => {
    const plan = planOf(chainOfFour());

    for (const buffer of plan.buffers) {
      expect(buffer.firstUse).toBeGreaterThanOrEqual(0);
      expect(buffer.lastUse).toBeGreaterThanOrEqual(buffer.firstUse);
      expect(buffer.lastUse).toBeLessThan(plan.steps);
    }
  });

  it('gives non-overlapping temporaries the same slot, so the peak stays under the total', () => {
    const plan = planOf(chainOfFour());

    expect(new Set(plan.buffers.map((b) => b.slot)).size).toBe(1);
    expect(plan.peakMemory).toBeLessThan(plan.totalBytesIfNeverShared);
  });

  it('gives temporaries that are live at the same time different slots', () => {
    const plan = planOf(twoLiveAtOnce());

    expect(plan.buffers).toHaveLength(2);
    const [first, second] = plan.buffers;
    expect(second.firstUse).toBeLessThanOrEqual(first.lastUse);
    expect(second.slot).not.toBe(first.slot);
    expect(plan.peakMemory).toBe(plan.totalBytesIfNeverShared);
  });

  it('names the buffer whose storage a reusing temporary took over', () => {
    const plan = planOf(chainOfFour());
    const byName = new Map(plan.buffers.map((b) => [b.name, b]));
    const reusers = plan.buffers.filter((b) => b.sharesWith !== null);

    expect(reusers.length).toBeGreaterThan(0);
    for (const reuser of reusers) {
      const donor = byName.get(reuser.sharesWith);
      expect(donor).toBeDefined();
      expect(donor.slot).toBe(reuser.slot);
      expect(donor.lastUse).toBeLessThanOrEqual(reuser.firstUse);
    }
  });

  it('totals every temporary it planned as the footprint sharing saved', () => {
    const plan = planOf(chainOfFour());
    const total = plan.buffers.reduce((sum, b) => sum + b.bytes, 0);

    expect(plan.totalBytesIfNeverShared).toBe(total);
    expect(plan.peakMemory).toBe(Math.max(...plan.buffers.map((b) => b.bytes)));
  });
});
