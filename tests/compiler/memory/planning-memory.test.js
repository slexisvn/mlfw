import { describe, it, expect } from 'vitest';
import { MemoryPlanner } from '../../../src/compiler/passes/memory/memory_planning.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IntImmNode, MathOpNode, VariableNode, BlockRealizeNode, AllocateNode
} from '../../../src/compiler/ir/tensor/nodes.js';

function makeVar(name) {
  return new VariableNode(name, 'int32');
}

function makeBind(name) {
  return new BlockRealizeNode(makeVar(name + '_v'), makeVar(name));
}

function makeBlock(name, reads, writes) {
  const binds = [makeBind('i')];
  const body = writes.length > 0
    ? new BufferStoreNode(writes[0].buffer, [], reads.length > 0 ? new BufferLoadNode(reads[0].buffer, []) : new IntImmNode(0))
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
