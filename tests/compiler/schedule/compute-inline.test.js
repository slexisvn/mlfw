import { describe, it, expect } from 'vitest';
import {
  ForNode, BlockNode, BufferStoreNode, BufferLoadNode, MathOpNode,
  VariableNode, IntImmNode, ForKind, SeqNode, PrimFunc,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { ScheduleValidator } from '../../../src/compiler/schedule/validator.js';
import { BackendPipeline } from '../../../src/backend/pipeline.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { spatialIter } from '../../_utils/ir_fixture.js';

function countLoads(node, name) {
  if (!node || typeof node !== 'object') return 0;
  let c = 0;
  if (node.type === 'BufferLoadNode' && node.buffer && node.buffer.name === name) c++;
  for (const k of ['a', 'b', 'expr', 'value', 'condition', 'thenBody', 'elseBody', 'body', 'initBody']) {
    if (node[k]) c += countLoads(node[k], name);
  }
  for (const arr of ['args', 'indices', 'stmts']) {
    if (node[arr]) for (const x of node[arr]) c += countLoads(x, name);
  }
  return c;
}

function buildProducerConsumer() {
  resetVarCounter();
  const A = new Buffer('A', [8], 'float32', 'global');
  const B = new Buffer('B', [8], 'float32', 'global');
  const C = new Buffer('C', [8], 'float32', 'global');

  const pi = new VariableNode('pi', 'int32');
  const pStore = new BufferStoreNode(B, [pi], new MathOpNode('*', new BufferLoadNode(A, [pi]), new IntImmNode(2)));
  const pBlock = new BlockNode('pb', [{ iterVar: pi, binding: pi }], [{ buffer: A }], [{ buffer: B }], pStore);
  const pNest = new ForNode(pi, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, pBlock);

  const ci = new VariableNode('ci', 'int32');
  const cStore = new BufferStoreNode(C, [ci], new MathOpNode('+', new BufferLoadNode(B, [ci]), new IntImmNode(1)));
  const cBlock = new BlockNode('cb', [{ iterVar: ci, binding: ci }], [{ buffer: B }], [{ buffer: C }], cStore);
  const cNest = new ForNode(ci, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, cBlock);

  return new Schedule(new PrimFunc('f', [], new SeqNode([pNest, cNest])));
}

describe('schedule: compute_inline / compute_at / reverse_compute_at', () => {
  it('inlines a producer into its consumer, eliminating loads of the intermediate', () => {
    const sch = buildProducerConsumer();
    expect(countLoads(sch.func.body, 'B')).toBe(1);
    expect(countLoads(sch.func.body, 'A')).toBe(1);

    sch.computeInline('pb');

    expect(countLoads(sch.func.body, 'B')).toBe(0);
    expect(countLoads(sch.func.body, 'A')).toBe(1);
    expect(sch.trace.length).toBe(1);
  });

  it('compute_at relocates a producer into a consumer loop (aligned case)', () => {
    const sch = buildProducerConsumer();
    expect(sch.getLoops('pb').map((l) => l.loopVar.name)).toEqual(['pi']);

    sch.computeAt('pb', 'ci');

    expect(sch.getLoops('pb').map((l) => l.loopVar.name)).toEqual(['ci']);
    expect(countLoads(sch.func.body, 'B')).toBe(1);
    expect(countLoads(sch.func.body, 'A')).toBe(1);
    expect(sch.trace.length).toBe(1);
  });

  it('reverse_compute_at relocates a consumer into a producer loop (aligned case)', () => {
    const sch = buildProducerConsumer();
    expect(sch.getLoops('cb').map((l) => l.loopVar.name)).toEqual(['ci']);

    sch.reverseComputeAt('cb', 'pi');

    expect(sch.getLoops('cb').map((l) => l.loopVar.name)).toEqual(['pi']);
    expect(countLoads(sch.func.body, 'B')).toBe(1);
    expect(sch.trace.length).toBe(1);
  });

  it('compute_at refuses non-aligned loop extents', () => {
    resetVarCounter();
    const A = new Buffer('A', [8], 'float32', 'global');
    const B = new Buffer('B', [8], 'float32', 'global');
    const C = new Buffer('C', [4], 'float32', 'global');
    const pi = new VariableNode('pi', 'int32');
    const pStore = new BufferStoreNode(B, [pi], new BufferLoadNode(A, [pi]));
    const pBlock = new BlockNode('pb', [{ iterVar: pi, binding: pi }], [{ buffer: A }], [{ buffer: B }], pStore);
    const pNest = new ForNode(pi, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, pBlock);
    const ci = new VariableNode('ci', 'int32');
    const cStore = new BufferStoreNode(C, [ci], new BufferLoadNode(B, [ci]));
    const cBlock = new BlockNode('cb', [{ iterVar: ci, binding: ci }], [{ buffer: B }], [{ buffer: C }], cStore);
    const cNest = new ForNode(ci, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, cBlock);
    const sch = new Schedule(new PrimFunc('f', [], new SeqNode([pNest, cNest])));
    expect(() => sch.computeAt('pb', 'ci')).toThrow(/iteration domain of 'pi' does not match 'ci'/);
  });

  it('computeInlineBlock inlines a multi-output producer and deletes the block', () => {
    resetVarCounter();
    const A = new Buffer('A', [8], 'float32', 'global');
    const P = new Buffer('P', [8], 'float32', 'local');
    const Q = new Buffer('Q', [8], 'float32', 'local');
    const OP = new Buffer('OP', [8], 'float32', 'global');
    const OQ = new Buffer('OQ', [8], 'float32', 'global');

    const pi = new VariableNode('pi', 'int32');
    const body = new SeqNode([
      new BufferStoreNode(P, [pi], new MathOpNode('*', new BufferLoadNode(A, [pi]), new IntImmNode(2))),
      new BufferStoreNode(Q, [pi], new MathOpNode('+', new BufferLoadNode(A, [pi]), new IntImmNode(1))),
    ]);
    const prod = new BlockNode('fuse', [{ iterVar: pi, binding: pi }], [{ buffer: A }], [{ buffer: P }, { buffer: Q }], body);
    const prodNest = new ForNode(pi, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, prod);

    const ci = new VariableNode('ci', 'int32');
    const cP = new BlockNode('cp', [{ iterVar: ci, binding: ci }], [{ buffer: P }], [{ buffer: OP }],
      new BufferStoreNode(OP, [ci], new BufferLoadNode(P, [ci])));
    const cPNest = new ForNode(ci, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, cP);
    const di = new VariableNode('di', 'int32');
    const cQ = new BlockNode('cq', [{ iterVar: di, binding: di }], [{ buffer: Q }], [{ buffer: OQ }],
      new BufferStoreNode(OQ, [di], new BufferLoadNode(Q, [di])));
    const cQNest = new ForNode(di, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, cQ);

    const sch = new Schedule(new PrimFunc('f', [], new SeqNode([prodNest, cPNest, cQNest])));
    expect(countLoads(sch.func.body, 'P')).toBe(1);
    expect(countLoads(sch.func.body, 'Q')).toBe(1);

    sch.computeInlineBlock('fuse');

    expect(countLoads(sch.func.body, 'P')).toBe(0);
    expect(countLoads(sch.func.body, 'Q')).toBe(0);
    expect(countLoads(sch.func.body, 'A')).toBe(2);
    expect(() => sch.getBlock('fuse')).toThrow();
  });

  it('refuses to inline a self-referential (recurrence) producer', () => {
    resetVarCounter();
    const B = new Buffer('B', [8], 'float32', 'global');
    const pi = new VariableNode('pi', 'int32');
    const store = new BufferStoreNode(B, [pi], new MathOpNode('+', new BufferLoadNode(B, [pi]), new IntImmNode(1)));
    const block = new BlockNode('pb', [{ iterVar: pi, binding: pi }], [{ buffer: B }], [{ buffer: B }], store);
    const nest = new ForNode(pi, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    const sch = new Schedule(new PrimFunc('f', [], nest));
    expect(() => sch.computeInline('pb')).toThrow(/self-referential|consumers/);
  });
});

describe('compute_inline inverts affine producer indices instead of requiring plain loop variables', () => {
  function flattenedProducer() {
    resetVarCounter();
    const A = new Buffer('A', [2, 4], 'float32', 'global');
    const B = new Buffer('B', [8], 'float32', 'global');
    const C = new Buffer('C', [8], 'float32', 'global');
    const i = new VariableNode('i', 'int32'), j = new VariableNode('j', 'int32');
    const vi = new VariableNode('vi', 'int32'), vj = new VariableNode('vj', 'int32');
    const flat = new MathOpNode('+', new MathOpNode('*', vi, new IntImmNode(4)), vj);
    const pStore = new BufferStoreNode(B, [flat], new MathOpNode('*', new BufferLoadNode(A, [vi, vj]), new IntImmNode(2)));
    const pBlock = new BlockNode('pb', [spatialIter(vi, i), spatialIter(vj, j)], [{ buffer: A }], [{ buffer: B }], pStore);
    const pNest = new ForNode(i, new IntImmNode(0), new IntImmNode(2), ForKind.SERIAL,
      new ForNode(j, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, pBlock));

    const c = new VariableNode('c', 'int32'), vc = new VariableNode('vc', 'int32');
    const cStore = new BufferStoreNode(C, [vc], new MathOpNode('+', new BufferLoadNode(B, [vc]), new IntImmNode(1)));
    const cBlock = new BlockNode('cb', [spatialIter(vc, c)], [{ buffer: B }], [{ buffer: C }], cStore);
    const cNest = new ForNode(c, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, cBlock);

    return new PrimFunc('f', [], new SeqNode([pNest, cNest]), new Map([['A', A], ['C', C]]));
  }

  const runCompiled = (func, input) => {
    const src = new BackendPipeline(CPUTarget()).compile(func).source;
    const out = new Float32Array(8).fill(NaN);
    new Function('return ' + src)()(input, out);
    return Array.from(out);
  };

  it('produces the same values as the un-inlined program', () => {
    const input = Float32Array.from({ length: 8 }, (_, k) => Math.sin(k * 1.7));
    const baseline = runCompiled(flattenedProducer(), input);

    const sch = new Schedule(flattenedProducer());
    sch.computeInline('pb');

    expect(countLoads(sch.func.body, 'B')).toBe(0);
    expect(ScheduleValidator.validate(sch.func)).toEqual([]);
    expect(runCompiled(sch.func, input)).toEqual(baseline);
  });

  it('refuses to inline a producer whose buffer feeds another access index', () => {
    resetVarCounter();
    const A = new Buffer('A', [8], 'float32', 'global');
    const B = new Buffer('B', [8], 'float32', 'global');
    const C = new Buffer('C', [8], 'float32', 'global');
    const pi = new VariableNode('pi', 'int32'), vpi = new VariableNode('vpi', 'int32');
    const pStore = new BufferStoreNode(B, [vpi], new BufferLoadNode(A, [vpi]));
    const pBlock = new BlockNode('pb', [spatialIter(vpi, pi)], [{ buffer: A }], [{ buffer: B }], pStore);
    const pNest = new ForNode(pi, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, pBlock);

    const ci = new VariableNode('ci', 'int32'), vci = new VariableNode('vci', 'int32');
    const cStore = new BufferStoreNode(C, [vci], new BufferLoadNode(A, [new BufferLoadNode(B, [vci])]));
    const cBlock = new BlockNode('cb', [spatialIter(vci, ci)], [{ buffer: A }, { buffer: B }], [{ buffer: C }], cStore);
    const cNest = new ForNode(ci, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, cBlock);

    const sch = new Schedule(new PrimFunc('f', [], new SeqNode([pNest, cNest]), new Map([['A', A], ['C', C]])));
    expect(() => sch.computeInline('pb')).toThrow(/indirect/);
  });

  it('refuses a producer whose write index is not an invertible affine map', () => {
    resetVarCounter();
    const A = new Buffer('A', [8], 'float32', 'global');
    const B = new Buffer('B', [8], 'float32', 'global');
    const C = new Buffer('C', [8], 'float32', 'global');
    const pi = new VariableNode('pi', 'int32'), vpi = new VariableNode('vpi', 'int32');
    const pStore = new BufferStoreNode(B, [new MathOpNode('*', vpi, new IntImmNode(2))], new BufferLoadNode(A, [vpi]));
    const pBlock = new BlockNode('pb', [spatialIter(vpi, pi)], [{ buffer: A }], [{ buffer: B }], pStore);
    const pNest = new ForNode(pi, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, pBlock);

    const ci = new VariableNode('ci', 'int32'), vci = new VariableNode('vci', 'int32');
    const cStore = new BufferStoreNode(C, [vci], new BufferLoadNode(B, [vci]));
    const cBlock = new BlockNode('cb', [spatialIter(vci, ci)], [{ buffer: B }], [{ buffer: C }], cStore);
    const cNest = new ForNode(ci, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, cBlock);

    const sch = new Schedule(new PrimFunc('f', [], new SeqNode([pNest, cNest]), new Map([['A', A], ['C', C]])));
    expect(() => sch.computeInline('pb')).toThrow(/not an invertible affine map/);
  });
});

describe('compute_at consults the block scope before moving a block', () => {
  it('refuses to move a producer across a block that depends on it', () => {
    resetVarCounter();
    const A = new Buffer('A', [8], 'float32', 'global');
    const B = new Buffer('B', [8], 'float32', 'global');
    const C = new Buffer('C', [8], 'float32', 'global');
    const D = new Buffer('D', [8], 'float32', 'global');
    const stage = (name, out, inp, lv, ivv) => {
      const store = new BufferStoreNode(out, [ivv], new MathOpNode('+', new BufferLoadNode(inp, [ivv]), new IntImmNode(1)));
      const block = new BlockNode(name, [spatialIter(ivv, lv)], [{ buffer: inp }], [{ buffer: out }], store);
      return new ForNode(lv, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
    };
    const n1 = stage('p', B, A, new VariableNode('i1', 'int32'), new VariableNode('v1', 'int32'));
    const n2 = stage('mid', C, B, new VariableNode('i2', 'int32'), new VariableNode('v2', 'int32'));
    const n3 = stage('last', D, C, new VariableNode('i3', 'int32'), new VariableNode('v3', 'int32'));
    const sch = new Schedule(new PrimFunc('f', [], new SeqNode([n1, n2, n3]), new Map([['A', A], ['D', D]])));

    expect(() => sch.computeAt('p', 'i3')).toThrow(/across 'mid' would violate a RAW dependence on buffer 'B'/);
  });

  it('allows moving a producer into the loop of its immediate consumer', () => {
    const sch = buildProducerConsumer();
    sch.computeAt('pb', 'ci');
    expect(sch.getLoops('pb').map((l) => l.loopVar.name)).toEqual(['ci']);
  });
});
