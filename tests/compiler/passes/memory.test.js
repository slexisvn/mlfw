import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, BlockRealizeNode,
  ForKind, AllocateNode
} from '../../../src/compiler/ir/tensor/nodes.js';
import {
  BufferLiveness, BufferInterval,
  InplaceAnalysis, InplaceCandidate,
  BufferAssignment, MemoryPool,
  MemoryPlanner
} from '../../../src/compiler/passes/memory/index.js';

function makeVar(name) { return new VariableNode(name, 'int32'); }
function intImm(v) { return new IntImmNode(v); }

function makeElementwiseBlock(name, reads, writes, loopVar) {
  const iv = makeVar(`v_${name}`);
  const bind = new BlockRealizeNode(iv, loopVar);
  const loadExprs = reads.map(buf => new BufferLoadNode(buf, [iv]));
  let expr = loadExprs[0];
  for (let i = 1; i < loadExprs.length; i++) {
    expr = new MathOpNode('+', expr, loadExprs[i]);
  }
  const store = new BufferStoreNode(writes[0], [iv], expr);
  return new BlockNode(
    name,
    [bind],
    reads.map(b => ({ buffer: b })),
    writes.map(b => ({ buffer: b })),
    store
  );
}

function makeChainFunc() {
  const A = new Buffer('A', [64], 'f32', 'global');
  const B = new Buffer('B', [64], 'f32', 'global');
  const T1 = new Buffer('T1', [64], 'f32', 'global');
  const T2 = new Buffer('T2', [64], 'f32', 'global');

  const i0 = makeVar('i0');
  const i1 = makeVar('i1');
  const i2 = makeVar('i2');

  const block0 = makeElementwiseBlock('b0', [A], [T1], i0);
  const block1 = makeElementwiseBlock('b1', [T1, B], [T2], i1);
  const block2 = makeElementwiseBlock('b2', [T2], [B], i2);

  const loop0 = new ForNode(i0, intImm(0), intImm(64), ForKind.SERIAL, block0);
  const loop1 = new ForNode(i1, intImm(0), intImm(64), ForKind.SERIAL, block1);
  const loop2 = new ForNode(i2, intImm(0), intImm(64), ForKind.SERIAL, block2);

  const body = new SeqNode([loop0, loop1, loop2]);

  const paramA = makeVar('param_A');
  const paramB = makeVar('param_B');
  const bufferMap = new Map([[paramA, A], [paramB, B]]);
  return new PrimFunc('chain', [paramA, paramB], body, bufferMap);
}

function makeDiamondFunc() {
  const A = new Buffer('A', [32], 'f32', 'global');
  const T1 = new Buffer('T1', [32], 'f32', 'global');
  const T2 = new Buffer('T2', [32], 'f32', 'global');
  const Out = new Buffer('Out', [32], 'f32', 'global');

  const i0 = makeVar('i0');
  const i1 = makeVar('i1');
  const i2 = makeVar('i2');

  const block0 = makeElementwiseBlock('exp_block', [A], [T1], i0);
  const block1 = makeElementwiseBlock('log_block', [A], [T2], i1);
  const block2 = makeElementwiseBlock('add_block', [T1, T2], [Out], i2);

  const body = new SeqNode([
    new ForNode(i0, intImm(0), intImm(32), ForKind.SERIAL, block0),
    new ForNode(i1, intImm(0), intImm(32), ForKind.SERIAL, block1),
    new ForNode(i2, intImm(0), intImm(32), ForKind.SERIAL, block2)
  ]);

  const pA = makeVar('pA');
  const pO = makeVar('pOut');
  return new PrimFunc('diamond', [pA, pO], body, new Map([[pA, A], [pO, Out]]));
}

describe('BufferLiveness', () => {
  it('computes live intervals for chain: A -> T1 -> T2 -> B', () => {
    const func = makeChainFunc();
    const result = BufferLiveness.analyze(func);

    assert.ok(result.intervals.has(func.bufferMap.get(func.params[0])));

    const t1Interval = result.intervals.get(
      [...result.intervals.keys()].find(b => b.name === 'T1')
    );
    assert.ok(t1Interval);
    assert.equal(t1Interval.firstUse, 0);
    assert.equal(t1Interval.lastUse, 1);

    const t2Interval = result.intervals.get(
      [...result.intervals.keys()].find(b => b.name === 'T2')
    );
    assert.ok(t2Interval);
    assert.equal(t2Interval.firstUse, 1);
    assert.equal(t2Interval.lastUse, 2);
  });

  it('T1 and T2 touch at stmt boundary in chain', () => {
    const func = makeChainFunc();
    const result = BufferLiveness.analyze(func);
    const t1 = [...result.intervals.keys()].find(b => b.name === 'T1');
    const t2 = [...result.intervals.keys()].find(b => b.name === 'T2');
    assert.ok(result.interfere(t1, t2));
  });

  it('T1 and T2 overlap in diamond', () => {
    const func = makeDiamondFunc();
    const result = BufferLiveness.analyze(func);
    const t1 = [...result.intervals.keys()].find(b => b.name === 'T1');
    const t2 = [...result.intervals.keys()].find(b => b.name === 'T2');
    assert.ok(result.interfere(t1, t2));
  });

  it('identifies param buffers vs temporaries', () => {
    const func = makeChainFunc();
    const result = BufferLiveness.analyze(func);
    const A = [...result.intervals.keys()].find(b => b.name === 'A');
    const T1 = [...result.intervals.keys()].find(b => b.name === 'T1');
    assert.ok(result.isParam(A));
    assert.ok(!result.isParam(T1));
  });

  it('getTemporaries returns only non-param buffers', () => {
    const func = makeChainFunc();
    const result = BufferLiveness.analyze(func);
    const temps = result.getTemporaries();
    assert.equal(temps.length, 2);
    assert.ok(temps.every(t => t.buffer.name === 'T1' || t.buffer.name === 'T2'));
  });
});

describe('InplaceAnalysis', () => {
  it('detects T1 can be reused by T2 in chain', () => {
    const func = makeChainFunc();
    const liveness = BufferLiveness.analyze(func);
    const candidates = InplaceAnalysis.analyze(func, liveness);
    assert.ok(candidates.length >= 1);
    assert.ok(candidates.some(c =>
      c.srcBuffer.name === 'T1' && c.dstBuffer.name === 'T2'
    ));
  });

  it('does not suggest in-place for overlapping T1/T2 in diamond', () => {
    const func = makeDiamondFunc();
    const liveness = BufferLiveness.analyze(func);
    const candidates = InplaceAnalysis.analyze(func, liveness);
    const bad = candidates.filter(c =>
      (c.srcBuffer.name === 'T1' && c.dstBuffer.name === 'T2') ||
      (c.srcBuffer.name === 'T2' && c.dstBuffer.name === 'T1')
    );
    assert.equal(bad.length, 0);
  });

  it('does not suggest in-place for param buffers', () => {
    const func = makeChainFunc();
    const liveness = BufferLiveness.analyze(func);
    const candidates = InplaceAnalysis.analyze(func, liveness);
    for (const c of candidates) {
      assert.ok(!liveness.isParam(c.srcBuffer));
      assert.ok(!liveness.isParam(c.dstBuffer));
    }
  });
});

describe('MemoryPool', () => {
  it('allocates with alignment', () => {
    const pool = new MemoryPool('global', 64);
    const buf1 = new Buffer('a', [16], 'f32', 'global');
    const b1 = pool.allocate(64, buf1);
    assert.equal(b1.offset, 0);
    assert.equal(b1.size, 64);

    const buf2 = new Buffer('b', [16], 'f32', 'global');
    const b2 = pool.allocate(64, buf2);
    assert.equal(b2.offset, 64);
  });

  it('reuses released memory', () => {
    const pool = new MemoryPool('global', 64);
    const buf1 = new Buffer('a', [16], 'f32', 'global');
    const b1 = pool.allocate(64, buf1);
    pool.release(b1);

    const buf2 = new Buffer('b', [16], 'f32', 'global');
    const b2 = pool.allocate(64, buf2);
    assert.equal(b2.offset, 0);
  });

  it('tracks peak usage', () => {
    const pool = new MemoryPool('global', 64);
    pool.allocate(256, new Buffer('a', [64], 'f32', 'global'));
    pool.allocate(128, new Buffer('b', [32], 'f32', 'global'));
    assert.equal(pool.peakUsage, 384);
  });
});

describe('BufferAssignment', () => {
  it('assigns non-overlapping buffers to same offset', () => {
    const buf1 = new Buffer('T1', [64], 'f32', 'global');
    const buf2 = new Buffer('T2', [64], 'f32', 'global');
    const intervals = [
      new BufferInterval(buf1, 0, 1, 'global'),
      new BufferInterval(buf2, 2, 3, 'global')
    ];
    const assignment = new BufferAssignment().assign(intervals);
    assert.equal(assignment.getOffset(buf1), 0);
    assert.equal(assignment.getOffset(buf2), 0);
  });

  it('assigns overlapping buffers to different offsets', () => {
    const buf1 = new Buffer('T1', [64], 'f32', 'global');
    const buf2 = new Buffer('T2', [64], 'f32', 'global');
    const intervals = [
      new BufferInterval(buf1, 0, 2, 'global'),
      new BufferInterval(buf2, 1, 3, 'global')
    ];
    const assignment = new BufferAssignment().assign(intervals);
    assert.notEqual(assignment.getOffset(buf1), assignment.getOffset(buf2));
  });

  it('handles in-place reuse', () => {
    const buf1 = new Buffer('T1', [64], 'f32', 'global');
    const buf2 = new Buffer('T2', [64], 'f32', 'global');
    const intervals = [
      new BufferInterval(buf1, 0, 1, 'global'),
      new BufferInterval(buf2, 2, 3, 'global')
    ];
    const candidates = [new InplaceCandidate(buf1, buf2, 'test')];
    const assignment = new BufferAssignment().assign(intervals, candidates);
    assert.equal(assignment.getOffset(buf1), assignment.getOffset(buf2));
    assert.equal(assignment.getAssignment(buf2).inplaceOf, buf1);
  });

  it('reports peak memory per scope', () => {
    const buf1 = new Buffer('T1', [64], 'f32', 'global');
    const buf2 = new Buffer('T2', [64], 'f32', 'shared');
    const intervals = [
      new BufferInterval(buf1, 0, 2, 'global'),
      new BufferInterval(buf2, 0, 2, 'shared')
    ];
    const assignment = new BufferAssignment().assign(intervals);
    assert.ok(assignment.peakMemory('global') > 0);
    assert.ok(assignment.peakMemory('shared') > 0);
    assert.ok(assignment.peakMemory() >= assignment.peakMemory('global'));
  });
});

describe('MemoryPlanner', () => {
  it('plans chain func with reuse', () => {
    const func = makeChainFunc();
    const planner = new MemoryPlanner({ alignment: 64 });
    const plan = planner.plan(func);

    assert.ok(plan.peakMemory() > 0);
    const report = plan.getReport();
    assert.equal(report.totalTemporaries, 2);
    assert.ok(report.peakMemory > 0);
  });

  it('plans diamond func without in-place for overlapping', () => {
    const func = makeDiamondFunc();
    const planner = new MemoryPlanner();
    const plan = planner.plan(func);
    const report = plan.getReport();
    assert.equal(report.totalTemporaries, 2);

    const t1 = [...plan.assignment.assignments.keys()].find(b => b.name === 'T1');
    const t2 = [...plan.assignment.assignments.keys()].find(b => b.name === 'T2');
    if (t1 && t2) {
      assert.notEqual(plan.assignment.getOffset(t1), plan.assignment.getOffset(t2));
    }
  });

  it('planAndRewrite inserts AllocateNodes', () => {
    const func = makeChainFunc();
    const planner = new MemoryPlanner();
    const { func: rewritten, plan } = planner.planAndRewrite(func);

    let hasAllocate = false;
    const checkAllocate = (node) => {
      if (!node) return;
      if (node.type === 'AllocateNode') { hasAllocate = true; return; }
      if (node.body) checkAllocate(node.body);
      if (node.stmts) node.stmts.forEach(checkAllocate);
    };
    checkAllocate(rewritten.body);
    assert.ok(hasAllocate);
  });

  it('disabling in-place still works', () => {
    const func = makeChainFunc();
    const planner = new MemoryPlanner({ enableInplace: false });
    const plan = planner.plan(func);
    assert.equal(plan.inplaceCandidates.length, 0);
    assert.ok(plan.peakMemory() > 0);
  });

  it('report scope breakdown', () => {
    const func = makeChainFunc();
    const plan = new MemoryPlanner().plan(func);
    const report = plan.getReport();
    assert.ok(report.scopeBreakdown.has('global'));
    const global = report.scopeBreakdown.get('global');
    assert.ok(global.peakUsage > 0);
    assert.ok(global.numBuffers > 0);
  });
});

describe('BufferInterval', () => {
  it('overlaps correctly', () => {
    const a = new BufferInterval(new Buffer('a', [1], 'f32', 'global'), 0, 5, 'global');
    const b = new BufferInterval(new Buffer('b', [1], 'f32', 'global'), 3, 8, 'global');
    const c = new BufferInterval(new Buffer('c', [1], 'f32', 'global'), 6, 10, 'global');
    assert.ok(a.overlaps(b));
    assert.ok(!a.overlaps(c));
    assert.ok(b.overlaps(c));
  });
});
