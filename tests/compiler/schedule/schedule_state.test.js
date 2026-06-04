import { describe, it, beforeEach } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  PrimFunc, ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, BlockRealizeNode, ForKind, MathOpNode, FloatImmNode
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { SRefTree } from '../../../src/compiler/schedule/sref.js';
import { DependencyAnalysis, DepKind } from '../../../src/compiler/schedule/dep_analysis.js';
import { ScheduleState } from '../../../src/compiler/schedule/schedule_state.js';

function makeMatmulFunc(M, N, K) {
  const A = new Buffer('A', [M, K], 'f32', 'global');
  const B = new Buffer('B', [K, N], 'f32', 'global');
  const C = new Buffer('C', [M, N], 'f32', 'global');
  const mi = new VariableNode('m', 'int32');
  const ni = new VariableNode('n', 'int32');
  const ki = new VariableNode('k', 'int32');
  const miv = new VariableNode('vm', 'int32');
  const niv = new VariableNode('vn', 'int32');
  const kiv = new VariableNode('vk', 'int32');
  const mBind = new BlockRealizeNode(miv, mi);
  const nBind = new BlockRealizeNode(niv, ni);
  const kBind = new BlockRealizeNode(kiv, ki);
  const loadA = new BufferLoadNode(A, [miv, kiv]);
  const loadB = new BufferLoadNode(B, [kiv, niv]);
  const loadC = new BufferLoadNode(C, [miv, niv]);
  const mul = new MathOpNode('*', loadA, loadB);
  const acc = new MathOpNode('+', loadC, mul);
  const store = new BufferStoreNode(C, [miv, niv], acc);
  const initStore = new BufferStoreNode(C, [miv, niv], new FloatImmNode(0));
  const block = new BlockNode('matmul', [mBind, nBind, kBind],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }], store, initStore);
  let body = block;
  body = new ForNode(ki, new IntImmNode(0), new IntImmNode(K), ForKind.SERIAL, body);
  body = new ForNode(ni, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, body);
  body = new ForNode(mi, new IntImmNode(0), new IntImmNode(M), ForKind.SERIAL, body);
  return new PrimFunc('matmul', [], body, new Map());
}

describe('SRefTree', () => {
  beforeEach(() => resetVarCounter());

  it('builds tree with correct structure', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const tree = new SRefTree(func);
    const blockSRef = tree.getBlockSRef('matmul');
    assert.ok(blockSRef);
    assert.ok(blockSRef.isBlock);
    const loops = tree.loopsOf('matmul');
    assert.equal(loops.length, 3);
    assert.ok(loops.every(l => l.isLoop));
  });

  it('block ancestors are the enclosing loops', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const tree = new SRefTree(func);
    const blockSRef = tree.getBlockSRef('matmul');
    const ancestors = blockSRef.loopAncestors();
    assert.equal(ancestors.length, 3);
  });

  it('allBlocks returns all blocks', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const tree = new SRefTree(func);
    const blocks = tree.allBlocks();
    assert.ok(blocks.length >= 1);
    assert.ok(blocks.some(b => b.node.name === 'matmul'));
  });

  it('allLoops returns all loops', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const tree = new SRefTree(func);
    const loops = tree.allLoops();
    assert.equal(loops.length, 3);
  });
});

describe('DependencyAnalysis', () => {
  beforeEach(() => resetVarCounter());

  it('detects read/write buffers for matmul', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const dep = new DependencyAnalysis(new SRefTree(func));
    const reads = dep.getReads('matmul');
    const writes = dep.getWrites('matmul');
    assert.ok(reads.includes('A'));
    assert.ok(reads.includes('B'));
    assert.ok(writes.includes('C'));
  });

  it('RAW dependency between producer and consumer blocks', async () => {
    const A = new Buffer('A', [8], 'f32', 'global');
    const B = new Buffer('B', [8], 'f32', 'global');
    const C = new Buffer('C', [8], 'f32', 'global');
    const i1 = new VariableNode('i', 'int32');
    const iv1 = new VariableNode('vi', 'int32');
    const bind1 = new BlockRealizeNode(iv1, i1);
    const store1 = new BufferStoreNode(B, [iv1], new BufferLoadNode(A, [iv1]));
    const block1 = new BlockNode('producer', [bind1], [{ buffer: A }], [{ buffer: B }], store1);

    const i2 = new VariableNode('j', 'int32');
    const iv2 = new VariableNode('vj', 'int32');
    const bind2 = new BlockRealizeNode(iv2, i2);
    const store2 = new BufferStoreNode(C, [iv2], new BufferLoadNode(B, [iv2]));
    const block2 = new BlockNode('consumer', [bind2], [{ buffer: B }], [{ buffer: C }], store2);

    const { SeqNode } = await import('../../../src/compiler/ir/tensor/nodes.js');
    let body1 = new ForNode(i1, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block1);
    let body2 = new ForNode(i2, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block2);
    const func = new PrimFunc('two_blocks', [], new SeqNode([body1, body2]), new Map());

    const dep = new DependencyAnalysis(new SRefTree(func));
    const deps = dep.computeDeps('producer', 'consumer');
    assert.ok(deps.some(d => d.kind === DepKind.RAW && d.bufferName === 'B'));
    assert.ok(!dep.canReorder('producer', 'consumer'));
  });
});

describe('ScheduleState', () => {
  beforeEach(() => resetVarCounter());

  it('provides block and loop bindings', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const state = new ScheduleState(func);
    const bb = state.getBlockBinding('matmul');
    assert.ok(bb);
    assert.ok(bb.readBuffers.includes('A'));
    assert.ok(bb.writeBuffers.includes('C'));
    assert.equal(bb.iterVars.length, 3);
  });

  it('tracks loop bindings with correct extents', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const state = new ScheduleState(func);
    const lb = state.getLoopBinding('m');
    assert.ok(lb);
    assert.equal(lb.extent, 64);
    assert.equal(lb.kind, ForKind.SERIAL);
  });

  it('invalidate rebuilds state after schedule transforms', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const sch = new Schedule(func);
    const loops = sch.getLoops('matmul');
    sch.split(loops[0], 16);

    const allVars = sch.state.allLoopVarNames();
    assert.ok(allVars.length > 3);
  });

  it('summary provides threadBindings', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const sch = new Schedule(func);
    const loops = sch.getLoops('matmul');
    const [mo, mi] = sch.split(loops[0], 16);
    sch.bindThread(mo, 'blockIdx.y');

    const summary = sch.state.summary();
    assert.ok(summary.threadBindings['blockIdx.y']);
    assert.equal(summary.gridDim[1], 4);
  });

  it('getReads/getWrites delegate to dep analysis', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const state = new ScheduleState(func);
    assert.deepEqual(state.getReads('matmul').sort(), ['A', 'B', 'C'].sort());
    assert.deepEqual(state.getWrites('matmul'), ['C']);
  });
});
