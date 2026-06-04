import { describe, it, beforeEach } from 'node:test';
import { strict as assert } from 'node:assert';
import {
  PrimFunc, ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, BlockRealizeNode, ForKind, SeqNode,
  MathOpNode, FloatImmNode
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleTrace } from '../../../src/compiler/schedule/trace.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';
import {
  SchedulePolicy, ElementwiseCPURule, ElementwiseGPURule,
  ReductionCPURule, ReductionGPURule, MatmulTiledCPURule,
  MatmulTiledGPURule, FallbackRule
} from '../../../src/compiler/schedule/rules.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { printTensorIR } from '../../../src/compiler/ir/tensor/printer.js';

function makeElementwiseFunc(shape) {
  const A = new Buffer('A', shape, 'f32', 'global');
  const B = new Buffer('B', shape, 'f32', 'global');
  const C = new Buffer('C', shape, 'f32', 'global');

  const loopVars = shape.map((_, i) => new VariableNode(`i${i}`, 'int32'));
  const iterVars = loopVars.map((v, i) => {
    const iv = new VariableNode(`v${i}`, 'int32');
    return new BlockRealizeNode(iv, v);
  });

  const indices = iterVars.map(b => b.iterVar);
  const loadA = new BufferLoadNode(A, indices);
  const loadB = new BufferLoadNode(B, indices);
  const add = new MathOpNode('+', loadA, loadB);
  const store = new BufferStoreNode(C, indices, add);

  const block = new BlockNode('add_block', iterVars,
    [{ buffer: A }, { buffer: B }],
    [{ buffer: C }],
    store
  );

  let body = block;
  for (let i = shape.length - 1; i >= 0; i--) {
    body = new ForNode(loopVars[i], new IntImmNode(0), new IntImmNode(shape[i]), ForKind.SERIAL, body);
  }

  return new PrimFunc('elementwise_add', [], body, new Map());
}

function makeReductionFunc(spatialSize, reduceSize) {
  const A = new Buffer('A', [spatialSize, reduceSize], 'f32', 'global');
  const initBuf = new Buffer('init', [], 'f32', 'global');
  const C = new Buffer('C', [spatialSize], 'f32', 'global');

  const si = new VariableNode('s0', 'int32');
  const ri = new VariableNode('r0', 'int32');
  const siv = new VariableNode('sv0', 'int32');
  const riv = new VariableNode('rv0', 'int32');
  const sBind = new BlockRealizeNode(siv, si);
  const rBind = new BlockRealizeNode(riv, ri);

  const loadA = new BufferLoadNode(A, [siv, riv]);
  const loadC = new BufferLoadNode(C, [siv]);
  const addExpr = new MathOpNode('+', loadC, loadA);
  const store = new BufferStoreNode(C, [siv], addExpr);
  const initStore = new BufferStoreNode(C, [siv], new FloatImmNode(0));

  const block = new BlockNode('reduce_block', [sBind, rBind],
    [{ buffer: A }],
    [{ buffer: C }],
    store, initStore
  );

  let body = block;
  body = new ForNode(ri, new IntImmNode(0), new IntImmNode(reduceSize), ForKind.SERIAL, body);
  body = new ForNode(si, new IntImmNode(0), new IntImmNode(spatialSize), ForKind.SERIAL, body);

  return new PrimFunc('reduce_sum', [], body, new Map());
}

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
    [{ buffer: A }, { buffer: B }],
    [{ buffer: C }],
    store, initStore
  );

  let body = block;
  body = new ForNode(ki, new IntImmNode(0), new IntImmNode(K), ForKind.SERIAL, body);
  body = new ForNode(ni, new IntImmNode(0), new IntImmNode(N), ForKind.SERIAL, body);
  body = new ForNode(mi, new IntImmNode(0), new IntImmNode(M), ForKind.SERIAL, body);

  return new PrimFunc('matmul', [], body, new Map());
}

describe('Schedule Primitives', () => {
  beforeEach(() => {
    resetVarCounter();
  });

  describe('split', () => {
    it('splits a loop into outer and inner', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      assert.equal(loops.length, 1);

      const [outer, inner] = sch.split(loops[0], 8);
      assert.equal(outer.loopVar.name.includes('_o'), true);
      assert.equal(inner.loopVar.name.includes('_i'), true);
      assert.equal(outer.extent.value, 8);
      assert.equal(inner.extent.value, 8);
    });

    it('handles non-divisible split with guard', () => {
      const func = makeElementwiseFunc([10]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      const [outer, inner] = sch.split(loops[0], 4);
      assert.equal(outer.extent.value, 3);
      assert.equal(inner.extent.value, 4);
      assert.equal(inner.body.type, 'IfThenElseNode');
    });

    it('rejects non-constant extent', () => {
      const varI = new VariableNode('i', 'int32');
      const dynExtent = new VariableNode('N', 'int32');
      const block = new BlockNode('b', [], [], [], null);
      const loop = new ForNode(varI, new IntImmNode(0), dynExtent, 'serial', block);
      const func = new PrimFunc('f', [], loop, new Map());
      const sch = new Schedule(func);
      assert.throws(() => sch.split(loop, 4), /non-constant extent/);
    });

    it('rejects invalid factor', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      assert.throws(() => sch.split(loops[0], 0));
      assert.throws(() => sch.split(loops[0], -1));
      assert.throws(() => sch.split(loops[0], 3.5));
    });

    it('records trace entry', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      sch.split(loops[0], 8);
      assert.equal(sch.trace.length, 1);
      assert.equal(sch.trace.steps[0].primitive, 'split');
    });
  });

  describe('reorder', () => {
    it('reorders two loops', () => {
      const func = makeElementwiseFunc([8, 16]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      assert.equal(loops.length, 2);

      sch.reorder(loops[1], loops[0]);

      const newLoops = sch.getLoops('add_block');
      assert.equal(newLoops[0].loopVar.name, 'i1');
      assert.equal(newLoops[1].loopVar.name, 'i0');
    });
  });

  describe('fuseLoops', () => {
    it('fuses two adjacent loops', () => {
      const func = makeElementwiseFunc([4, 8]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      const fused = sch.fuseLoops(loops[0], loops[1]);
      assert.equal(fused.extent.value, 32);
      assert.ok(fused.loopVar.name.includes('fused'));
    });

    it('rejects non-adjacent loops', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const varA = new VariableNode('a', 'int32');
      const varB = new VariableNode('b', 'int32');
      const loopB = new ForNode(varB, new IntImmNode(0), new IntImmNode(4), 'serial', null);
      const loopA = new ForNode(varA, new IntImmNode(0), new IntImmNode(4), 'serial',
        new SeqNode([loopB])
      );
      const f = new PrimFunc('f', [], loopA, new Map());
      const sch2 = new Schedule(f);
      assert.throws(() => sch2.fuseLoops(loopA, loopB), /direct child/);
    });
  });

  describe('tile', () => {
    it('tiles a 2D loop nest', () => {
      const func = makeElementwiseFunc([64, 64]);
      const sch = new Schedule(func);
      const { outerLoops, innerLoops } = sch.tile('add_block', [0, 1], [8, 8]);
      assert.equal(outerLoops.length, 2);
      assert.equal(innerLoops.length, 2);
    });
  });

  describe('vectorize', () => {
    it('marks loop as vectorized', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      const [_, inner] = sch.split(loops[0], 8);
      sch.vectorize(inner);
      assert.equal(inner.kind, ForKind.VECTORIZED);
    });
  });

  describe('unroll', () => {
    it('marks loop as unrolled', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      sch.unroll(loops[0]);
      assert.equal(loops[0].kind, ForKind.UNROLLED);
    });
  });

  describe('parallelize', () => {
    it('marks loop as parallel', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      sch.parallelize(loops[0]);
      assert.equal(loops[0].kind, ForKind.PARALLEL);
    });
  });

  describe('bindThread', () => {
    it('binds loop to thread axis', () => {
      const func = makeElementwiseFunc([256]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      const [outer, inner] = sch.split(loops[0], 32);
      sch.bindThread(outer, 'blockIdx.x');
      sch.bindThread(inner, 'threadIdx.x');
      assert.equal(outer.kind, ForKind.THREAD_BINDING);
      assert.equal(outer.threadTag, 'blockIdx.x');
      assert.equal(inner.kind, ForKind.THREAD_BINDING);
      assert.equal(inner.threadTag, 'threadIdx.x');
    });

    it('rejects invalid thread tag', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      assert.throws(() => sch.bindThread(loops[0], 'invalid'), /Invalid thread tag/);
    });
  });

  describe('cacheRead', () => {
    it('creates a cache buffer and read stage', () => {
      const func = makeElementwiseFunc([8]);
      const sch = new Schedule(func);
      const cacheBuf = sch.cacheRead('add_block', 'A', 'shared');
      assert.equal(cacheBuf.name, 'A_shared');
      assert.equal(cacheBuf.scope, 'shared');

      const block = sch.getBlock('add_block');
      assert.ok(block.reads.some(r => r.buffer.name === 'A_shared'));
    });

    it('rejects unknown buffer', () => {
      const func = makeElementwiseFunc([8]);
      const sch = new Schedule(func);
      assert.throws(() => sch.cacheRead('add_block', 'Z', 'shared'), /does not read/);
    });
  });

  describe('cacheWrite', () => {
    it('creates a cache buffer and writeback stage', () => {
      const func = makeElementwiseFunc([8]);
      const sch = new Schedule(func);
      const cacheBuf = sch.cacheWrite('add_block', 'C', 'local');
      assert.equal(cacheBuf.name, 'C_local');
      assert.equal(cacheBuf.scope, 'local');

      const block = sch.getBlock('add_block');
      assert.ok(block.writes.some(r => r.buffer.name === 'C_local'));
    });
  });

  describe('annotate', () => {
    it('attaches annotation to loop', () => {
      const func = makeElementwiseFunc([64]);
      const sch = new Schedule(func);
      const loops = sch.getLoops('add_block');
      sch.annotate(loops[0], 'pragma_unroll', 4);
      assert.equal(loops[0].annotations.pragma_unroll, 4);
    });
  });
});

describe('ScheduleTrace', () => {
  beforeEach(() => {
    resetVarCounter();
  });

  it('records and serializes schedule steps', () => {
    const func = makeElementwiseFunc([64]);
    const sch = new Schedule(func);
    const loops = sch.getLoops('add_block');
    sch.split(loops[0], 8);
    const newLoops = sch.getLoops('add_block');
    sch.vectorize(newLoops[newLoops.length - 1]);

    const trace = sch.getTrace();
    assert.equal(trace.length, 2);

    const serialized = trace.serialize();
    assert.equal(serialized.length, 2);
    assert.equal(serialized[0].primitive, 'split');
    assert.equal(serialized[1].primitive, 'vectorize');
  });

  it('deserializes trace', () => {
    const data = [
      { primitive: 'split', args: ['i0', 8] },
      { primitive: 'vectorize', args: ['i0_1_i'] }
    ];
    const trace = ScheduleTrace.deserialize(data);
    assert.equal(trace.length, 2);
    assert.equal(trace.steps[0].primitive, 'split');
    assert.equal(trace.steps[1].primitive, 'vectorize');
  });
});

describe('ScheduleValidator', () => {
  beforeEach(() => {
    resetVarCounter();
  });

  it('validates clean schedule with no errors', () => {
    const func = makeElementwiseFunc([64]);
    const sch = new Schedule(func);
    const loops = sch.getLoops('add_block');
    sch.split(loops[0], 8);
    const errors = sch.verify();
    assert.equal(errors.length, 0);
  });

  it('detects duplicate thread bindings', () => {
    const func = makeElementwiseFunc([64, 64]);
    const sch = new Schedule(func);
    const loops = sch.getLoops('add_block');
    sch.bindThread(loops[0], 'threadIdx.x');
    sch.bindThread(loops[1], 'threadIdx.x');

    const errors = sch.verify();
    assert.ok(errors.some(e => e.includes('Duplicate thread binding')));
  });

  it('detects missing threadTag on thread-bound loop', () => {
    const func = makeElementwiseFunc([64]);
    const sch = new Schedule(func);
    const loops = sch.getLoops('add_block');
    loops[0].kind = ForKind.THREAD_BINDING;
    loops[0].threadTag = null;

    const errors = sch.verify();
    assert.ok(errors.some(e => e.includes('missing threadTag')));
  });
});

describe('Schedule Rules', () => {
  beforeEach(() => {
    resetVarCounter();
  });

  describe('ElementwiseCPURule', () => {
    it('parallelizes outer loop and vectorizes inner', () => {
      const func = makeElementwiseFunc([64]);
      const target = CPUTarget({ vectorWidth: 8 });
      const sch = new Schedule(func);
      const rule = new ElementwiseCPURule();

      assert.ok(rule.matches(func, 'add_block', target));
      rule.apply(sch, 'add_block', target);

      const loops = sch.getLoops('add_block');
      assert.ok(loops.some(l => l.kind === ForKind.PARALLEL));
      assert.ok(loops.some(l => l.kind === ForKind.VECTORIZED));
    });

    it('does not match GPU target', () => {
      const func = makeElementwiseFunc([64]);
      const target = GPUTarget();
      const rule = new ElementwiseCPURule();
      assert.equal(rule.matches(func, 'add_block', target), false);
    });
  });

  describe('ElementwiseGPURule', () => {
    it('binds loops to GPU threads', () => {
      const func = makeElementwiseFunc([1024]);
      const target = GPUTarget();
      const sch = new Schedule(func);
      const rule = new ElementwiseGPURule();

      assert.ok(rule.matches(func, 'add_block', target));
      rule.apply(sch, 'add_block', target);

      const loops = sch.getLoops('add_block');
      assert.ok(loops.some(l => l.kind === ForKind.THREAD_BINDING && l.threadTag === 'blockIdx.x'));
      assert.ok(loops.some(l => l.kind === ForKind.THREAD_BINDING && l.threadTag === 'threadIdx.x'));
    });

    it('handles small extents with single thread block', () => {
      const func = makeElementwiseFunc([128]);
      const target = GPUTarget();
      const sch = new Schedule(func);
      const rule = new ElementwiseGPURule();
      rule.apply(sch, 'add_block', target);

      const loops = sch.getLoops('add_block');
      assert.ok(loops.some(l => l.kind === ForKind.THREAD_BINDING));
    });
  });

  describe('ReductionCPURule', () => {
    it('parallelizes spatial loop of reduction', () => {
      const func = makeReductionFunc(64, 128);
      const target = CPUTarget();
      const sch = new Schedule(func);
      const rule = new ReductionCPURule();

      assert.ok(rule.matches(func, 'reduce_block', target));
      rule.apply(sch, 'reduce_block', target);

      const loops = sch.getLoops('reduce_block');
      assert.ok(loops[0].kind === ForKind.PARALLEL);
    });
  });

  describe('ReductionGPURule', () => {
    it('binds spatial loop to GPU threads', () => {
      const func = makeReductionFunc(1024, 128);
      const target = GPUTarget();
      const sch = new Schedule(func);
      const rule = new ReductionGPURule();

      assert.ok(rule.matches(func, 'reduce_block', target));
      rule.apply(sch, 'reduce_block', target);

      const loops = sch.getLoops('reduce_block');
      assert.ok(loops.some(l => l.kind === ForKind.THREAD_BINDING));
    });
  });

  describe('MatmulTiledCPURule', () => {
    it('tiles M dimension of matmul', () => {
      const func = makeMatmulFunc(128, 128, 64);
      const target = CPUTarget();
      const sch = new Schedule(func);
      const rule = new MatmulTiledCPURule();

      assert.ok(rule.matches(func, 'matmul', target));
      rule.apply(sch, 'matmul', target);

      const loops = sch.getLoops('matmul');
      assert.ok(loops.some(l => l.kind === ForKind.PARALLEL));
      assert.ok(loops.length > 3);
    });
  });

  describe('MatmulTiledGPURule', () => {
    it('tiles and binds matmul to GPU grid', () => {
      const func = makeMatmulFunc(256, 256, 64);
      const target = GPUTarget();
      const sch = new Schedule(func);
      const rule = new MatmulTiledGPURule();

      assert.ok(rule.matches(func, 'matmul', target));
      rule.apply(sch, 'matmul', target);

      const loops = sch.getLoops('matmul');
      const threadBound = loops.filter(l => l.kind === ForKind.THREAD_BINDING);
      assert.ok(threadBound.length >= 2);
    });
  });

  describe('FallbackRule', () => {
    it('always matches', () => {
      const func = makeElementwiseFunc([64]);
      const rule = new FallbackRule();
      assert.ok(rule.matches(func, 'add_block', CPUTarget()));
      assert.ok(rule.matches(func, 'add_block', GPUTarget()));
    });

    it('parallelizes outer loop on CPU', () => {
      const func = makeElementwiseFunc([64]);
      const target = CPUTarget();
      const sch = new Schedule(func);
      const rule = new FallbackRule();
      rule.apply(sch, 'add_block', target);
      const loops = sch.getLoops('add_block');
      assert.equal(loops[0].kind, ForKind.PARALLEL);
    });
  });
});

describe('SchedulePolicy', () => {
  beforeEach(() => {
    resetVarCounter();
  });

  it('selects correct rule for elementwise CPU', () => {
    const func = makeElementwiseFunc([64]);
    const target = CPUTarget({ vectorWidth: 8 });
    const policy = new SchedulePolicy(target);
    const rule = policy.selectRule(func, 'add_block');
    assert.ok(rule instanceof ElementwiseCPURule);
  });

  it('selects correct rule for elementwise GPU', () => {
    const func = makeElementwiseFunc([64]);
    const target = GPUTarget();
    const policy = new SchedulePolicy(target);
    const rule = policy.selectRule(func, 'add_block');
    assert.ok(rule instanceof ElementwiseGPURule);
  });

  it('selects matmul rule for matmul blocks', () => {
    const func = makeMatmulFunc(128, 128, 64);
    const target = CPUTarget();
    const policy = new SchedulePolicy(target);
    const rule = policy.selectRule(func, 'matmul');
    assert.ok(rule instanceof MatmulTiledCPURule);
  });

  it('applies to all blocks in a function', () => {
    const func = makeElementwiseFunc([64, 64]);
    const target = CPUTarget({ vectorWidth: 8 });
    const policy = new SchedulePolicy(target);
    const sch = new Schedule(func);
    const applied = policy.applyToAllBlocks(sch);

    assert.ok(applied.size > 0);
    assert.ok(applied.has('add_block'));
  });
});

describe('Target', () => {
  it('CPU target has expected defaults', () => {
    const cpu = CPUTarget();
    assert.ok(cpu.isCPU());
    assert.ok(!cpu.isGPU());
    assert.ok(cpu.supportsVectorization());
    assert.ok(!cpu.supportsThreadBinding());
    assert.equal(cpu.vectorWidth, 8);
  });

  it('GPU target has expected defaults', () => {
    const gpu = GPUTarget();
    assert.ok(gpu.isGPU());
    assert.ok(!gpu.isCPU());
    assert.ok(gpu.supportsThreadBinding());
    assert.equal(gpu.warpSize, 32);
    assert.equal(gpu.maxThreadsPerBlock, 1024);
    assert.ok(gpu.hasLibraryOp('dot'));
  });

  it('supports override', () => {
    const cpu = CPUTarget({ vectorWidth: 16, numCores: 32 });
    assert.equal(cpu.vectorWidth, 16);
    assert.equal(cpu.numCores, 32);
  });
});

describe('Compound Schedule Scenarios', () => {
  beforeEach(() => {
    resetVarCounter();
  });

  it('split + reorder + vectorize pipeline', () => {
    const func = makeElementwiseFunc([64, 64]);
    const sch = new Schedule(func);

    let loops = sch.getLoops('add_block');
    const [io, ii] = sch.split(loops[1], 8);

    loops = sch.getLoops('add_block');
    assert.equal(loops.length, 3);

    sch.vectorize(ii);

    const errors = sch.verify();
    assert.equal(errors.length, 0);
  });

  it('GPU-style elementwise: fuse + split + bind', () => {
    const func = makeElementwiseFunc([16, 32]);
    const sch = new Schedule(func);

    let loops = sch.getLoops('add_block');
    const fused = sch.fuseLoops(loops[0], loops[1]);
    assert.equal(fused.extent.value, 512);

    const [bx, tx] = sch.split(fused, 128);
    sch.bindThread(bx, 'blockIdx.x');
    sch.bindThread(tx, 'threadIdx.x');

    const errors = sch.verify();
    assert.equal(errors.length, 0);
  });

  it('tiled matmul with thread binding', () => {
    const func = makeMatmulFunc(64, 64, 32);
    const sch = new Schedule(func);

    let loops = sch.getLoops('matmul');
    assert.equal(loops.length, 3);

    const [mo, mi] = sch.split(loops[0], 16);
    sch.bindThread(mo, 'blockIdx.y');

    loops = sch.getLoops('matmul');
    const nLoop = loops.find(l => l.loopVar.name === 'n');
    if (nLoop) {
      const [no, ni] = sch.split(nLoop, 16);
      sch.bindThread(no, 'blockIdx.x');
    }

    const errors = sch.verify();
    assert.equal(errors.length, 0);
  });

  it('end-to-end: schedule + print produces valid IR', () => {
    const func = makeElementwiseFunc([256]);
    const sch = new Schedule(func);
    const loops = sch.getLoops('add_block');
    const [outer, inner] = sch.split(loops[0], 32);
    sch.parallelize(outer);
    sch.vectorize(inner);

    const ir = printTensorIR(func);
    assert.ok(ir.includes('@parallel'));
    assert.ok(ir.includes('@vectorized'));
    assert.ok(ir.includes('add_block'));
  });
});
