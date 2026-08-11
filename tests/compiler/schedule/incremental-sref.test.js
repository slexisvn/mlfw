import { describe, it, expect, beforeEach } from 'vitest';
import {
  PrimFunc, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode, ForNode,
  VariableNode, IntImmNode, MathOpNode, ForKind,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';
import { SRefTree } from '../../../src/compiler/schedule/sref.js';

function iv(name) { return new VariableNode(name, 'int32'); }

function elementwiseBlock(name, buf, indices) {
  return new BlockNode(name, indices.map(v => ({ iterVar: v, binding: v })),
    [], [{ buffer: buf }], new BufferStoreNode(buf, indices, new IntImmNode(0)));
}

function loopNest(vars, extents, inner) {
  let body = inner;
  for (let i = vars.length - 1; i >= 0; i--) {
    body = new ForNode(vars[i], new IntImmNode(0), new IntImmNode(extents[i]), ForKind.SERIAL, body);
  }
  return body;
}

function matmulFunc(name, M, N, K) {
  const A = new Buffer('A', [M, K], 'f32', 'global');
  const B = new Buffer('B', [K, N], 'f32', 'global');
  const C = new Buffer('C', [M, N], 'f32', 'global');
  const m = iv('m'), n = iv('n'), k = iv('k');
  const prod = new MathOpNode('*', new BufferLoadNode(A, [m, k]), new BufferLoadNode(B, [k, n]));
  const acc = new MathOpNode('+', new BufferLoadNode(C, [m, n]), prod);
  const block = new BlockNode(name,
    [{ iterVar: m, binding: m }, { iterVar: n, binding: n }, { iterVar: k, binding: k }],
    [{ buffer: A }, { buffer: B }], [{ buffer: C }],
    new BufferStoreNode(C, [m, n], acc), new BufferStoreNode(C, [m, n], new IntImmNode(0)));
  let nest = block;
  for (const [v, e] of [[k, K], [n, N], [m, M]]) {
    nest = new ForNode(v, new IntImmNode(0), new IntImmNode(e), ForKind.SERIAL, nest);
  }
  return new PrimFunc(name, [], nest, new Map([['A', A], ['B', B], ['C', C]]));
}

function blockNames(tree) { return tree.allBlocks().map(s => s.node.name).sort(); }
function loopSet(tree) { return new Set(tree.allLoops().map(s => s.node)); }

function expectIncrementalMatchesRebuild(sch) {
  const inc = sch.state.tree;
  const fresh = new SRefTree(sch.func);

  expect(blockNames(inc)).toEqual(blockNames(fresh));

  const il = loopSet(inc), fl = loopSet(fresh);
  expect(il.size).toBe(fl.size);
  for (const node of fl) expect(il.has(node)).toBe(true);

  for (const name of blockNames(fresh)) {
    const incB = inc.getBlockSRef(name);
    const freshB = fresh.getBlockSRef(name);
    expect(incB).toBeTruthy();
    expect(incB.node).toBe(freshB.node);
    expect(inc.loopsOf(name).map(s => s.node)).toEqual(fresh.loopsOf(name).map(s => s.node));
  }
}

describe('incremental SRefTree stays identical to a full rebuild after each primitive', () => {
  beforeEach(() => resetVarCounter());

  it('split', () => {
    const buf = new Buffer('A', [16], 'f32', 'global');
    const i = iv('i');
    const func = new PrimFunc('f', [], loopNest([i], [16], elementwiseBlock('b', buf, [i])));
    const sch = new Schedule(func);
    sch.split(sch.getLoops('b')[0], 4);
    expect(sch.getLoops('b').length).toBe(2);
    expectIncrementalMatchesRebuild(sch);
  });

  it('split with a guard (non-divisible factor)', () => {
    const buf = new Buffer('A', [10], 'f32', 'global');
    const i = iv('i');
    const func = new PrimFunc('f', [], loopNest([i], [10], elementwiseBlock('b', buf, [i])));
    const sch = new Schedule(func);
    sch.split(sch.getLoops('b')[0], 4);
    expectIncrementalMatchesRebuild(sch);
  });

  it('reorder', () => {
    const buf = new Buffer('A', [4, 8], 'f32', 'global');
    const i = iv('i'), j = iv('j');
    const func = new PrimFunc('f', [], loopNest([i, j], [4, 8], elementwiseBlock('b', buf, [i, j])));
    const sch = new Schedule(func);
    const loops = sch.getLoops('b');
    sch.reorder(loops[1], loops[0]);
    expect(sch.getLoops('b').map(l => l.loopVar.name)).toEqual(['j', 'i']);
    expectIncrementalMatchesRebuild(sch);
  });

  it('fuseLoops', () => {
    const buf = new Buffer('A', [4, 4], 'f32', 'global');
    const i = iv('i'), j = iv('j');
    const func = new PrimFunc('f', [], loopNest([i, j], [4, 4], elementwiseBlock('b', buf, [i, j])));
    const sch = new Schedule(func);
    const loops = sch.getLoops('b');
    sch.fuseLoops(loops[0], loops[1]);
    expect(sch.getLoops('b').length).toBe(1);
    expectIncrementalMatchesRebuild(sch);
  });

  it('tile (split + reorder chain)', () => {
    const func = matmulFunc('g', 32, 32, 32);
    const sch = new Schedule(func);
    sch.tile('g', [0, 1], [8, 8]);
    expectIncrementalMatchesRebuild(sch);
  });

  it('decomposeReduction (creates init/upd blocks)', () => {
    const func = matmulFunc('g', 4, 4, 8);
    const sch = new Schedule(func);
    sch.decomposeReduction('g');
    expect(blockNames(sch.state.tree)).toEqual(['g_init', 'g_upd']);
    expectIncrementalMatchesRebuild(sch);
  });

  it('rfactor (creates rf_p/rf_c blocks)', () => {
    const func = matmulFunc('g', 4, 4, 8);
    const sch = new Schedule(func);
    sch.rfactor('g', 'k', 2);
    expect(blockNames(sch.state.tree)).toEqual(['g_rf_c', 'g_rf_p']);
    expectIncrementalMatchesRebuild(sch);
  });

  it('blockize (wraps a loop in a new block)', () => {
    const buf = new Buffer('A', [4, 8], 'f32', 'global');
    const i = iv('i'), j = iv('j');
    const func = new PrimFunc('f', [], loopNest([i, j], [4, 8], elementwiseBlock('b', buf, [i, j])));
    const sch = new Schedule(func);
    sch.blockize(sch.getLoops('b')[0]);
    expectIncrementalMatchesRebuild(sch);
  });

  it('scheduling one block leaves a sibling block untouched in the tree', () => {
    const a = new Buffer('A', [16], 'f32', 'global');
    const c = new Buffer('C', [16], 'f32', 'global');
    const i = iv('i'), j = iv('j');
    const body = new SeqNode([
      loopNest([i], [16], elementwiseBlock('ba', a, [i])),
      loopNest([j], [16], elementwiseBlock('bc', c, [j])),
    ]);
    const sch = new Schedule(new PrimFunc('f', [], body));

    const siblingLoopsBefore = sch.getLoops('bc').map(l => l.loopVar.name);
    const siblingNodeBefore = sch.getBlock('bc');

    sch.split(sch.getLoops('ba')[0], 4);

    expect(sch.getBlock('bc')).toBe(siblingNodeBefore);
    expect(sch.getLoops('bc').map(l => l.loopVar.name)).toEqual(siblingLoopsBefore);
    expectIncrementalMatchesRebuild(sch);
  });

  it('a chain of primitives keeps the tree consistent throughout', () => {
    const func = matmulFunc('g', 16, 16, 16);
    const sch = new Schedule(func);
    sch.split(sch.getLoops('g')[0], 4);
    expectIncrementalMatchesRebuild(sch);
    sch.split(sch.getLoops('g')[1], 2);
    expectIncrementalMatchesRebuild(sch);
    const loops = sch.getLoops('g');
    sch.reorder(loops[1], loops[0]);
    expectIncrementalMatchesRebuild(sch);
  });
});
