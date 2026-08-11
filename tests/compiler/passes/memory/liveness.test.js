import { describe, it, expect } from 'vitest';
import { BufferLiveness, BufferInterval, BufferLivenessResult } from '../../../../src/compiler/passes/memory/buffer_liveness.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  IntImmNode, MathOpNode, ForKind, BlockRealizeNode, VariableNode,
  IfThenElseNode, AllocateNode
} from '../../../../src/compiler/ir/tensor/nodes.js';

function makeVar(name) {
  return new VariableNode(name, 'int32');
}

function makeBind(name) {
  const v = makeVar(name);
  return new BlockRealizeNode(makeVar(name + '_v'), v);
}

function makeSimpleBlock(name, reads, writes) {
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

describe('BufferInterval', () => {
  it('overlaps returns true for intersecting intervals', () => {
    const buf1 = new Buffer('a', [4], 'f32', 'global');
    const buf2 = new Buffer('b', [4], 'f32', 'global');
    const a = new BufferInterval(buf1, 0, 5, 'global');
    const b = new BufferInterval(buf2, 3, 8, 'global');
    expect(a.overlaps(b)).toBe(true);
    expect(b.overlaps(a)).toBe(true);
  });

  it('overlaps returns false for non-intersecting intervals', () => {
    const buf1 = new Buffer('a', [4], 'f32', 'global');
    const buf2 = new Buffer('b', [4], 'f32', 'global');
    const a = new BufferInterval(buf1, 0, 3, 'global');
    const b = new BufferInterval(buf2, 5, 8, 'global');
    expect(a.overlaps(b)).toBe(false);
    expect(b.overlaps(a)).toBe(false);
  });

  it('overlaps returns true for adjacent intervals sharing an endpoint', () => {
    const buf1 = new Buffer('a', [4], 'f32', 'global');
    const buf2 = new Buffer('b', [4], 'f32', 'global');
    const a = new BufferInterval(buf1, 0, 5, 'global');
    const b = new BufferInterval(buf2, 5, 8, 'global');
    expect(a.overlaps(b)).toBe(true);
  });

  it('size returns correct byte count', () => {
    const buf = new Buffer('a', [4, 8], 'f32', 'global');
    const interval = new BufferInterval(buf, 0, 1, 'global');
    expect(interval.size).toBe(4 * 8 * 4);
  });
});

describe('BufferLiveness.analyze', () => {
  it('single block touches all read/write buffers', () => {
    const inBuf = new Buffer('in', [4], 'f32', 'global');
    const outBuf = new Buffer('out', [4], 'f32', 'global');
    const block = makeSimpleBlock('b0', [{ buffer: inBuf }], [{ buffer: outBuf }]);
    const pf = makePrimFunc(block, [inBuf, outBuf]);

    const result = BufferLiveness.analyze(pf);
    expect(result.intervals.has(inBuf)).toBe(true);
    expect(result.intervals.has(outBuf)).toBe(true);
    expect(result.stmtOrder.length).toBe(1);
  });

  it('sequential blocks assign increasing stmt indices', () => {
    const a = new Buffer('a', [4], 'f32', 'global');
    const b = new Buffer('b', [4], 'f32', 'global');
    const c = new Buffer('c', [4], 'f32', 'global');
    const block0 = makeSimpleBlock('b0', [{ buffer: a }], [{ buffer: b }]);
    const block1 = makeSimpleBlock('b1', [{ buffer: b }], [{ buffer: c }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [a]);

    const result = BufferLiveness.analyze(pf);
    expect(result.stmtOrder.length).toBe(2);
    expect(result.stmtOrder[0].idx).toBeLessThan(result.stmtOrder[1].idx);
  });

  it('temporary buffer not in paramBuffers is detected by getTemporaries', () => {
    const paramBuf = new Buffer('param', [4], 'f32', 'global');
    const tempBuf = new Buffer('temp', [4], 'f32', 'global');
    const outBuf = new Buffer('out', [4], 'f32', 'global');
    const block0 = makeSimpleBlock('b0', [{ buffer: paramBuf }], [{ buffer: tempBuf }]);
    const block1 = makeSimpleBlock('b1', [{ buffer: tempBuf }], [{ buffer: outBuf }]);
    const seq = new SeqNode([block0, block1]);
    const pf = makePrimFunc(seq, [paramBuf, outBuf]);

    const result = BufferLiveness.analyze(pf);
    const temps = result.getTemporaries();
    expect(temps.length).toBe(1);
    expect(temps[0].buffer).toBe(tempBuf);
  });

  it('isParam returns true for buffers in the bufferMap', () => {
    const paramBuf = new Buffer('param', [4], 'f32', 'global');
    const block = makeSimpleBlock('b0', [{ buffer: paramBuf }], []);
    const pf = makePrimFunc(block, [paramBuf]);

    const result = BufferLiveness.analyze(pf);
    expect(result.isParam(paramBuf)).toBe(true);
  });

  it('isParam returns false for temporary buffers', () => {
    const paramBuf = new Buffer('param', [4], 'f32', 'global');
    const tempBuf = new Buffer('temp', [4], 'f32', 'global');
    const block = makeSimpleBlock('b0', [{ buffer: paramBuf }], [{ buffer: tempBuf }]);
    const pf = makePrimFunc(block, [paramBuf]);

    const result = BufferLiveness.analyze(pf);
    expect(result.isParam(tempBuf)).toBe(false);
  });

  it('liveness interval spans from first to last use across blocks', () => {
    const a = new Buffer('a', [4], 'f32', 'global');
    const t = new Buffer('t', [4], 'f32', 'global');
    const b = new Buffer('b', [4], 'f32', 'global');
    const c = new Buffer('c', [4], 'f32', 'global');
    const block0 = makeSimpleBlock('b0', [{ buffer: a }], [{ buffer: t }]);
    const block1 = makeSimpleBlock('b1', [{ buffer: a }], [{ buffer: b }]);
    const block2 = makeSimpleBlock('b2', [{ buffer: t }], [{ buffer: c }]);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [a]);

    const result = BufferLiveness.analyze(pf);
    const tInterval = result.intervals.get(t);
    expect(tInterval.firstUse).toBe(0);
    expect(tInterval.lastUse).toBe(2);
  });

  it('interfere returns true for overlapping temporaries', () => {
    const a = new Buffer('a', [4], 'f32', 'global');
    const t1 = new Buffer('t1', [4], 'f32', 'global');
    const t2 = new Buffer('t2', [4], 'f32', 'global');
    const out = new Buffer('out', [4], 'f32', 'global');
    const block0 = makeSimpleBlock('b0', [{ buffer: a }], [{ buffer: t1 }]);
    const block1 = makeSimpleBlock('b1', [{ buffer: a }], [{ buffer: t2 }]);
    const binds = [makeBind('i')];
    const body = new BufferStoreNode(out, [], new MathOpNode('+', new BufferLoadNode(t1, []), new BufferLoadNode(t2, [])));
    const block2 = new BlockNode('b2', binds, [{ buffer: t1 }, { buffer: t2 }], [{ buffer: out }], body);
    const seq = new SeqNode([block0, block1, block2]);
    const pf = makePrimFunc(seq, [a, out]);

    const result = BufferLiveness.analyze(pf);
    expect(result.interfere(t1, t2)).toBe(true);
  });

  it('interfere returns false for non-overlapping temporaries', () => {
    const a = new Buffer('a', [4], 'f32', 'global');
    const t1 = new Buffer('t1', [4], 'f32', 'global');
    const t2 = new Buffer('t2', [4], 'f32', 'global');
    const out = new Buffer('out', [4], 'f32', 'global');
    const block0 = makeSimpleBlock('b0', [{ buffer: a }], [{ buffer: t1 }]);
    const block1 = makeSimpleBlock('b1', [{ buffer: t1 }], [{ buffer: out }]);
    const block2 = makeSimpleBlock('b2', [{ buffer: a }], [{ buffer: t2 }]);
    const block3 = makeSimpleBlock('b3', [{ buffer: t2 }], [{ buffer: out }]);
    const seq = new SeqNode([block0, block1, block2, block3]);
    const pf = makePrimFunc(seq, [a, out]);

    const result = BufferLiveness.analyze(pf);
    expect(result.interfere(t1, t2)).toBe(false);
  });

  it('blocks inside ForNode are analyzed', () => {
    const buf = new Buffer('buf', [4], 'f32', 'global');
    const block = makeSimpleBlock('inner', [{ buffer: buf }], [{ buffer: buf }]);
    const loop = new ForNode(makeVar('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc(loop, [buf]);

    const result = BufferLiveness.analyze(pf);
    expect(result.intervals.has(buf)).toBe(true);
    expect(result.stmtOrder.length).toBe(1);
  });

  it('buffer touched early in a multi-statement loop body stays live to loop end', () => {
    const early = new Buffer('early', [4], 'f32', 'global');
    const mid = new Buffer('mid', [4], 'f32', 'global');
    const tail = new Buffer('tail', [4], 'f32', 'global');
    const inBuf = new Buffer('in', [4], 'f32', 'global');
    const innerA = makeSimpleBlock('ia', [{ buffer: inBuf }], [{ buffer: early }]);
    const innerB = makeSimpleBlock('ib', [{ buffer: inBuf }], [{ buffer: mid }]);
    const innerC = makeSimpleBlock('ic', [{ buffer: inBuf }], [{ buffer: tail }]);
    const body = new SeqNode([innerA, innerB, innerC]);
    const loop = new ForNode(makeVar('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, body);
    const pf = makePrimFunc(loop, [inBuf]);

    const result = BufferLiveness.analyze(pf);
    const earlyInterval = result.intervals.get(early);
    expect(earlyInterval.firstUse).toBe(0);
    expect(earlyInterval.lastUse).toBe(2);
  });

  it('liveness extends to the join across If branches', () => {
    const a = new Buffer('a', [4], 'f32', 'global');
    const t1 = new Buffer('t1', [4], 'f32', 'global');
    const t2 = new Buffer('t2', [4], 'f32', 'global');
    const out = new Buffer('out', [4], 'f32', 'global');
    const pre = makeSimpleBlock('pre', [{ buffer: a }], [{ buffer: t1 }]);
    const thenBlock = makeSimpleBlock('then', [{ buffer: t1 }], [{ buffer: t2 }]);
    const elseBlock = makeSimpleBlock('else', [{ buffer: a }], [{ buffer: out }]);
    const ite = new IfThenElseNode(new IntImmNode(1), thenBlock, elseBlock);
    const seq = new SeqNode([pre, ite]);
    const pf = makePrimFunc(seq, [a, out]);

    const result = BufferLiveness.analyze(pf);
    const t1Interval = result.intervals.get(t1);
    expect(t1Interval.firstUse).toBe(0);
    expect(t1Interval.lastUse).toBe(2);
  });

  it('blocks inside IfThenElseNode are analyzed', () => {
    const bufA = new Buffer('a', [4], 'f32', 'global');
    const bufB = new Buffer('b', [4], 'f32', 'global');
    const thenBlock = makeSimpleBlock('then', [{ buffer: bufA }], [{ buffer: bufB }]);
    const elseBlock = makeSimpleBlock('else', [{ buffer: bufA }], [{ buffer: bufB }]);
    const ite = new IfThenElseNode(new IntImmNode(1), thenBlock, elseBlock);
    const pf = makePrimFunc(ite, [bufA, bufB]);

    const result = BufferLiveness.analyze(pf);
    expect(result.stmtOrder.length).toBe(2);
  });
});
