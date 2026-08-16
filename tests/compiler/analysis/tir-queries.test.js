import { describe, it, expect } from 'vitest';
import {
  PrimFunc, ForNode, ForKind, SeqNode, AllocateNode, LetStmtNode, VecCopyNode,
  BlockNode, BlockRealizeNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, FloatImmNode, MathOpNode, CallExternNode,
  IfThenElseNode, CompareNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { walkScoped } from '../../../src/compiler/ir/ir_visitor.js';
import {
  varNamesOf, usesAnyVar, usesAnyVarIn, storedBufferNames, referencedBuffers,
  allocatedBufferNames, collectScopeBindings, findLoopOfKind, staticExtentOf,
  bufferAccessCount, vectorValueDtype, loadsAreUnitStrideIn, hasBufferAccessMatching,
  indicesAreLaneAffine, loopCarriedDependenceIn, guardedLoadsAreInRange,
} from '../../../src/compiler/analysis/tir_queries.js';

const v = (name) => new VariableNode(name, 'int32');
const i32 = (n) => new IntImmNode(n);
const f32v = (n) => new FloatImmNode(n);
const mul = (a, b) => new MathOpNode('*', a, b);
const buf = (name, shape, dtype = 'float32') => new Buffer(name, shape, dtype, 'global');
const serial = (name, extent, body) => new ForNode(v(name), i32(0), i32(extent), ForKind.SERIAL, body);

function primFunc(body, storage = []) {
  const map = new Map();
  for (const b of storage) map.set(v(b.name), b);
  return new PrimFunc('f', [], body, map);
}

describe('variable queries', () => {
  it('collects variable names from a whole expression tree', () => {
    expect([...varNamesOf(mul(v('i'), mul(v('j'), i32(4))))].sort()).toEqual(['i', 'j']);
  });

  it('answers dependency questions without materialising the name set', () => {
    const expr = mul(v('i'), i32(4));
    expect(usesAnyVar(expr, new Set(['i']))).toBe(true);
    expect(usesAnyVar(expr, new Set(['k']))).toBe(false);
    expect(usesAnyVar(expr, new Set())).toBe(false);
    expect(usesAnyVarIn([i32(0), expr], new Set(['i']))).toBe(true);
    expect(usesAnyVarIn([i32(0), i32(1)], new Set(['i']))).toBe(false);
  });
});

describe('buffer queries reach every schema field, not a hardcoded key list', () => {
  it('finds a load that only appears inside a loop extent', () => {
    const n = buf('n', [1]);
    const out = buf('out', [8]);
    const body = new ForNode(v('i'), i32(0), new BufferLoadNode(n, [i32(0)]), ForKind.SERIAL,
      new BufferStoreNode(out, [v('i')], f32v(1)));

    expect([...referencedBuffers(body).keys()].sort()).toEqual(['n', 'out']);
    expect(bufferAccessCount(body)).toBe(2);
  });

  it('finds a load that only appears inside a store index', () => {
    const idx = buf('idx', [8]);
    const out = buf('out', [8]);
    const body = new BufferStoreNode(out, [new BufferLoadNode(idx, [i32(0)])], f32v(1));

    expect([...referencedBuffers(body).keys()].sort()).toEqual(['idx', 'out']);
  });

  it('counts a VecCopy destination as a stored buffer', () => {
    const src = buf('src', [8]);
    const dst = buf('dst', [8]);
    const body = new VecCopyNode(dst, [i32(0)], src, [i32(0)], 4);
    expect([...storedBufferNames(body)]).toEqual(['dst']);
  });

  it('collects allocations at any nesting depth', () => {
    const t = buf('t', [4]);
    const u = buf('u', [4]);
    const body = new AllocateNode(t, 'local', serial('i', 4, new AllocateNode(u, 'local', new SeqNode([]))));
    expect([...allocatedBufferNames(body)].sort()).toEqual(['t', 'u']);
  });

  it('reports the value-position dtype when every value access agrees', () => {
    const a = buf('a', [4], 'float16');
    const b = buf('b', [4], 'float16');
    const body = new SeqNode([
      new BufferStoreNode(a, [i32(0)], f32v(1)),
      new BufferStoreNode(b, [i32(0)], f32v(1)),
    ]);
    expect(vectorValueDtype(body, 'float32')).toBe('float16');
    expect(vectorValueDtype(new SeqNode([]), 'float32')).toBe('float32');
  });

  it('reports null when value-position accesses mix dtypes', () => {
    const half = buf('h', [4], 'float16');
    const single = buf('s', [4], 'float32');
    const body = new SeqNode([
      new BufferStoreNode(half, [i32(0)], f32v(1)),
      new BufferStoreNode(single, [i32(0)], f32v(1)),
    ]);
    expect(vectorValueDtype(body, 'float32')).toBe(null);
  });

  it('accepts index expressions that are lane-affine', () => {
    const out = buf('out', [8]);
    const src = buf('src', [8]);
    const lanes = new Set(['i']);
    const body = new BufferStoreNode(out, [v('i')], new BufferLoadNode(src, [new MathOpNode('+', v('i'), i32(2))]));
    expect(indicesAreLaneAffine(body, lanes)).toBe(true);
  });

  it('rejects an indirect index that varies per lane', () => {
    const out = buf('out', [8]);
    const src = buf('src', [8]);
    const perm = buf('perm', [8], 'int32');
    const lanes = new Set(['i']);
    const gathered = new BufferLoadNode(src, [new BufferLoadNode(perm, [v('i')])]);
    expect(indicesAreLaneAffine(new BufferStoreNode(out, [v('i')], gathered), lanes)).toBe(false);
  });

  it('rejects a scaled index that skips elements between lanes', () => {
    const out = buf('out', [8]);
    const lanes = new Set(['i']);
    const body = new BufferStoreNode(out, [mul(v('i'), i32(2))], f32v(1));
    expect(indicesAreLaneAffine(body, lanes)).toBe(false);
  });

  it('detects a load from a stored buffer at a different index', () => {
    const acc = buf('acc', [8]);
    const shifted = new BufferLoadNode(acc, [new MathOpNode('+', v('i'), i32(1))]);
    expect(loopCarriedDependenceIn(new BufferStoreNode(acc, [v('i')], shifted))).toBe(true);
  });

  it('allows a read-modify-write of the same element', () => {
    const acc = buf('acc', [8]);
    const same = new BufferLoadNode(acc, [v('i')]);
    expect(loopCarriedDependenceIn(new BufferStoreNode(acc, [v('i')], same))).toBe(false);
  });

  it('ignores index-position accesses of a different dtype', () => {
    const data = buf('data', [4], 'float32');
    const out = buf('out', [4], 'float32');
    const idx = buf('idx', [4], 'int32');
    const gathered = new BufferLoadNode(data, [new BufferLoadNode(idx, [v('i')])]);
    const body = new BufferStoreNode(out, [v('i')], gathered);
    expect(vectorValueDtype(body, 'float32')).toBe('float32');
  });

  it('matches buffer accesses by dtype through both node families', () => {
    const half = buf('h', [4], 'float16');
    const single = buf('s', [4], 'float32');
    const isHalf = (dtype) => dtype === 'float16';
    expect(hasBufferAccessMatching(new BufferStoreNode(half, [i32(0)], f32v(1)), isHalf)).toBe(true);
    expect(hasBufferAccessMatching(new BufferStoreNode(single, [i32(0)], f32v(1)), isHalf)).toBe(false);
  });
});

describe('scope bindings', () => {
  it('collects the iteration variables a block binds', () => {
    const out = buf('out', [8]);
    const inner = new BlockNode('b1', [new BlockRealizeNode(v('vj'), mul(v('i'), i32(2)))], [], [],
      new BufferStoreNode(out, [v('vj')], f32v(1)));
    const body = serial('i', 4, new BlockNode('b0', [new BlockRealizeNode(v('vi'), v('i'))], [], [], inner));

    const bindings = collectScopeBindings(body);
    expect(bindings.map(b => b.name)).toEqual(['vi', 'vj']);
    expect(usesAnyVar(bindings[1].expr, new Set(['i']))).toBe(true);
    expect(usesAnyVar(bindings[0].expr, new Set(['k']))).toBe(false);
  });

  it('keeps a let-bound variable reachable through the value expression', () => {
    const out = buf('out', [8]);
    const body = serial('i', 4, new LetStmtNode(v('j'), mul(v('i'), i32(2)),
      new BufferStoreNode(out, [v('j')], f32v(1))));
    expect(usesAnyVar(body, new Set(['i']))).toBe(true);
    expect(collectScopeBindings(body)).toEqual([]);
  });
});

describe('loop queries', () => {
  it('finds the first loop of a kind in program order', () => {
    const out = buf('out', [8]);
    const body = new SeqNode([
      new ForNode(v('a'), i32(0), i32(3), ForKind.PARALLEL, new BufferStoreNode(out, [v('a')], f32v(1))),
      new ForNode(v('b'), i32(0), i32(5), ForKind.PARALLEL, new BufferStoreNode(out, [v('b')], f32v(2))),
    ]);
    const loop = findLoopOfKind(body, ForKind.PARALLEL);
    expect(loop.loopVar.name).toBe('a');
    expect(staticExtentOf(loop)).toBe(3);
    expect(findLoopOfKind(body, ForKind.VECTORIZED)).toBe(null);
  });

  it('reports zero for a dynamic extent', () => {
    const out = buf('out', [8]);
    const dyn = new ForNode(v('a'), i32(0), v('n'), ForKind.SERIAL, new BufferStoreNode(out, [v('a')], f32v(1)));
    expect(staticExtentOf(dyn)).toBe(0);
  });
});

describe('unit-stride load detection', () => {
  it('rejects a load whose leading index moves with the vector lane', () => {
    const a = buf('a', [8, 8]);
    const out = buf('out', [8]);
    const strided = new BufferStoreNode(out, [v('k')], new BufferLoadNode(a, [v('k'), i32(0)]));
    const contiguous = new BufferStoreNode(out, [v('k')], new BufferLoadNode(a, [i32(0), v('k')]));
    expect(loadsAreUnitStrideIn(strided, new Set(['k']))).toBe(false);
    expect(loadsAreUnitStrideIn(contiguous, new Set(['k']))).toBe(true);
  });

  it('treats an empty lane-variable set as trivially unit stride', () => {
    const a = buf('a', [8, 8]);
    const out = buf('out', [8]);
    const strided = new BufferStoreNode(out, [v('k')], new BufferLoadNode(a, [v('k'), i32(0)]));
    expect(loadsAreUnitStrideIn(strided, new Set())).toBe(true);
  });
});

describe('walkScoped', () => {
  it('threads a caller-defined scope down the tree and lets a visitor prune', () => {
    const out = buf('out', [8]);
    const body = serial('i', 4, serial('j', 4, new BufferStoreNode(out, [v('i')], f32v(1))));

    const depths = [];
    walkScoped(body, 0, (node, depth) => {
      if (node.type === 'ForNode') depths.push(depth);
      return depth + 1;
    });
    expect(depths).toEqual([0, 1]);

    const seen = [];
    walkScoped(body, 0, (node, depth) => {
      seen.push(node.type);
      return node.type === 'ForNode' && depth === 0 ? false : depth + 1;
    });
    expect(seen).toEqual(['ForNode']);
  });
});

describe('storedBufferNames on a whole function', () => {
  it('separates written buffers from read-only ones', () => {
    const a = buf('a', [8]);
    const out = buf('out', [8]);
    const fn = primFunc(serial('i', 8, new BufferStoreNode(out, [v('i')], new BufferLoadNode(a, [v('i')]))), [a, out]);
    expect([...storedBufferNames(fn.body)]).toEqual(['out']);
    expect(bufferAccessCount(fn.body)).toBe(2);
  });

  it('does not miss stores hidden inside an extern call argument', () => {
    const out = buf('out', [8]);
    const a = buf('a', [8]);
    const body = new BufferStoreNode(out, [i32(0)],
      new CallExternNode('max', [new BufferLoadNode(a, [i32(0)]), f32v(0)], 'float32'));
    expect([...referencedBuffers(body).keys()].sort()).toEqual(['a', 'out']);
  });
});

describe('guarded loads under a vectorized select', () => {
  const laneVars = new Set(['j']);

  it('accepts branches that load the index the loop stores to', () => {
    const out = buf('out', [4, 8]);
    const a = buf('a', [4, 8]);
    const b = buf('b', [4, 8]);
    const idx = [v('i'), v('j')];
    const select = new IfThenElseNode(
      new CompareNode('gt', new BufferLoadNode(a, idx), f32v(0)),
      new BufferLoadNode(a, [v('i'), v('j')]),
      new BufferLoadNode(b, [v('i'), v('j')]),
    );
    expect(guardedLoadsAreInRange(new BufferStoreNode(out, idx, select), laneVars)).toBe(true);
  });

  it('rejects a branch that loads a shifted index the guard was protecting', () => {
    const out = buf('out', [4, 8]);
    const src = buf('src', [4, 6]);
    const shifted = new BufferLoadNode(src, [v('i'), new MathOpNode('-', v('j'), i32(1))]);
    const select = new IfThenElseNode(new CompareNode('ge', v('j'), i32(1)), shifted, f32v(0));
    expect(guardedLoadsAreInRange(new BufferStoreNode(out, [v('i'), v('j')], select), laneVars)).toBe(false);
  });

  it('rejects a guarded load even when the guard itself is lane-invariant', () => {
    const out = buf('out', [6, 8]);
    const src = buf('src', [4, 8]);
    const shiftedRow = new BufferLoadNode(src, [new MathOpNode('-', v('i'), i32(1)), v('j')]);
    const select = new IfThenElseNode(new CompareNode('ge', v('i'), i32(1)), shiftedRow, f32v(0));
    expect(guardedLoadsAreInRange(new BufferStoreNode(out, [v('i'), v('j')], select), laneVars)).toBe(false);
  });
});
