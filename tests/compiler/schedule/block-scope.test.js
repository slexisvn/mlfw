import { describe, it, expect } from 'vitest';
import {
  ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, MathOpNode, ForKind, PrimFunc,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { spatialIter } from '../../_utils/ir_fixture.js';
import { ScheduleState } from '../../../src/compiler/schedule/schedule_state.js';
import { DepKind } from '../../../src/compiler/analysis/dependence.js';

const v = (name) => new VariableNode(name, 'int32');
const i32 = (n) => new IntImmNode(n);

function elementwiseBlock(name, loopVar, iterVar, extent, outBuf, outIdx, inBuf, inIdx, reads) {
  const store = new BufferStoreNode(outBuf, outIdx, new MathOpNode('+', new BufferLoadNode(inBuf, inIdx), i32(1)));
  const block = new BlockNode(name, [spatialIter(iterVar, loopVar)], reads, [{ buffer: outBuf }], store);
  return new ForNode(loopVar, i32(0), i32(extent), ForKind.SERIAL, block);
}

function producerConsumer() {
  const A = new Buffer('A', [8], 'float32', 'global');
  const B = new Buffer('B', [8], 'float32', 'global');
  const C = new Buffer('C', [8], 'float32', 'global');
  const pi = v('pi'), vpi = v('vpi');
  const ci = v('ci'), vci = v('vci');
  const producer = elementwiseBlock('p', pi, vpi, 8, B, [vpi], A, [vpi], [{ buffer: A }]);
  const consumer = elementwiseBlock('c', ci, vci, 8, C, [vci], B, [vci], [{ buffer: B }]);
  return new PrimFunc('f', [], new SeqNode([producer, consumer]), new Map([['A', A], ['C', C]]));
}

const scopeOf = (state) => state.scopes.get(null);

describe('BlockScope dependency graph over the sref tree', () => {
  it('derives a RAW edge from producer to consumer through the shared buffer', () => {
    const state = new ScheduleState(producerConsumer());
    const scope = scopeOf(state);
    const p = state.tree.getBlockSRef('p');
    const c = state.tree.getBlockSRef('c');

    const raw = scope.depsBySrc(p).filter((d) => d.kind === DepKind.RAW);
    expect(raw).toHaveLength(1);
    expect(raw[0].dst).toBe(c);
    expect(raw[0].buffer.name).toBe('B');
    expect(scope.consumersOf(p)).toEqual([c]);
    expect(scope.producersOf(c)).toEqual([p]);
  });

  it('reports no edge between blocks that touch disjoint regions of the same buffer', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const O = new Buffer('O', [8], 'float32', 'global');
    const lo = v('lo'), vlo = v('vlo');
    const hi = v('hi'), vhi = v('vhi');
    const lowHalf = new BlockNode('low', [spatialIter(vlo, lo)], [{ buffer: A }], [{ buffer: O }],
      new BufferStoreNode(O, [vlo], new BufferLoadNode(A, [vlo])));
    const highHalf = new BlockNode('high', [spatialIter(vhi, new MathOpNode('+', hi, i32(4)))], [{ buffer: O }], [{ buffer: A }],
      new BufferStoreNode(A, [vhi], new BufferLoadNode(O, [vhi])));
    const body = new SeqNode([
      new ForNode(lo, i32(0), i32(4), ForKind.SERIAL, lowHalf),
      new ForNode(hi, i32(0), i32(4), ForKind.SERIAL, highHalf),
    ]);
    const state = new ScheduleState(new PrimFunc('f', [], body, new Map([['A', A], ['O', O]])));
    const scope = scopeOf(state);
    expect(scope.deps).toHaveLength(0);
  });

  it('classifies a write-after-read edge and drops stage_pipeline for the scope', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const T = new Buffer('T', [8], 'float32', 'global');
    const ri = v('ri'), vri = v('vri');
    const wi = v('wi'), vwi = v('vwi');
    const reader = elementwiseBlock('reader', ri, vri, 8, T, [vri], A, [vri], [{ buffer: A }]);
    const writer = elementwiseBlock('writer', wi, vwi, 8, A, [vwi], T, [vwi], [{ buffer: T }]);
    const state = new ScheduleState(new PrimFunc('f', [], new SeqNode([reader, writer]), new Map([['A', A]])));
    const scope = scopeOf(state);

    const kinds = scope.deps.map((d) => d.kind).sort();
    expect(kinds).toContain(DepKind.WAR);
    expect(scope.stagePipeline).toBe(false);
  });

  it('marks region_cover false when the producer writes less than the consumer reads', () => {
    const A = new Buffer('A', [8], 'float32', 'global');
    const B = new Buffer('B', [8], 'float32', 'global');
    const C = new Buffer('C', [8], 'float32', 'global');
    const pi = v('pi'), vpi = v('vpi');
    const ci = v('ci'), vci = v('vci');
    const producer = elementwiseBlock('p', pi, vpi, 4, B, [vpi], A, [vpi], [{ buffer: A }]);
    const consumer = elementwiseBlock('c', ci, vci, 8, C, [vci], B, [vci], [{ buffer: B }]);
    const state = new ScheduleState(new PrimFunc('f', [], new SeqNode([producer, consumer]), new Map([['A', A], ['C', C]])));
    const scope = scopeOf(state);

    expect(scope.blockInfo(state.tree.getBlockSRef('c')).regionCover).toBe(false);
    expect(scope.blockInfo(state.tree.getBlockSRef('p')).regionCover).toBe(true);
    expect(scope.stagePipeline).toBe(false);
  });

  it('a full-coverage producer/consumer pair is a stage pipeline with region cover on every block', () => {
    const state = new ScheduleState(producerConsumer());
    const scope = scopeOf(state);
    for (const sref of scope.children) expect(scope.blockInfo(sref).regionCover).toBe(true);
    expect(scope.stagePipeline).toBe(true);
  });

  it('reports affine_binding per block and rebuilds the scope after a tree mutation', () => {
    const state = new ScheduleState(producerConsumer());
    const first = state.scopes;
    expect(state.blockInfo(state.tree.getBlockSRef('p')).affineBinding).toBe(true);

    state.invalidate();
    expect(state.scopes).not.toBe(first);
    expect(state.blockInfo(state.tree.getBlockSRef('p')).affineBinding).toBe(true);
  });
});
