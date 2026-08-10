import { describe, it, expect } from 'vitest';
import {
  PrimFunc, SeqNode, ForNode, BlockNode, BufferStoreNode, BufferLoadNode,
  IfThenElseNode, VariableNode, IntImmNode, CompareNode, ForKind, mathOp,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import { analyzeStorageRequirements, buffersRequiringDefinedStorage, StorageRequirement } from '../../../src/compiler/analysis/buffer_dataflow.js';
import { collectBufferAccesses } from '../../../src/compiler/analysis/buffer_access.js';

const v = (n) => new VariableNode(n, 'int32');
const i32 = (n) => new IntImmNode(n);

function loopOver(name, extent, makeBody) {
  const iv = v(name);
  return new ForNode(iv, i32(0), i32(extent), ForKind.SERIAL, makeBody(iv));
}

function fn(body, buffers) {
  const map = new Map();
  for (const b of buffers) map.set(b.name, b);
  return new PrimFunc('f', [], body, map);
}

function requirementOf(primFunc, buffer) {
  return analyzeStorageRequirements(primFunc).get(buffer).requirement;
}

describe('buffer access collection resolves index ranges', () => {
  it('binds ForNode loop vars to their [min, extent]', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const body = loopOver('i', 8, (i) => new BufferStoreNode(T, [i], i32(0)));
    const { byBuffer } = collectBufferAccesses(body);
    expect(byBuffer.get(T)[0].regions).toEqual([[0, 8]]);
  });

  it('resolves BlockNode iterVars through their loop-var bindings', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const vi = v('vi');
    const body = loopOver('i', 8, (i) => new BlockNode(
      'blk', [{ iterVar: vi, binding: i }], [], [{ buffer: T }],
      new BufferStoreNode(T, [vi], i32(0)),
    ));
    const { byBuffer } = collectBufferAccesses(body);
    expect(byBuffer.get(T)[0].regions).toEqual([[0, 8]]);
  });

  it('leaves a non-affine index unresolved rather than guessing', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const IDX = new Buffer('IDX', [8], 'i32', 'global');
    const body = loopOver('i', 8, (i) => new BufferStoreNode(T, [new BufferLoadNode(IDX, [i])], i32(1)));
    const { byBuffer } = collectBufferAccesses(body);
    expect(byBuffer.get(T)[0].regions).toEqual([null]);
  });

  it('restores an outer binding after an inner loop shadows the same name', () => {
    const T = new Buffer('T', [4], 'f32', 'global');
    const inner = loopOver('i', 2, (i) => new BufferStoreNode(T, [i], i32(0)));
    const outer = loopOver('i', 4, (i) => new SeqNode([inner, new BufferStoreNode(T, [i], i32(1))]));
    const { byBuffer } = collectBufferAccesses(outer);
    const regions = byBuffer.get(T).map((a) => a.regions);
    expect(regions).toEqual([[[0, 2]], [[0, 4]]]);
  });
});

describe('storage requirement: buffers fully defined before first read are FRESH', () => {
  it('accepts a loop nest that covers the whole buffer', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const O = new Buffer('O', [8], 'f32', 'global');
    const body = new SeqNode([
      loopOver('i', 8, (i) => new BufferStoreNode(T, [i], i32(3))),
      loopOver('j', 8, (j) => new BufferStoreNode(O, [j], new BufferLoadNode(T, [j]))),
    ]);
    expect(requirementOf(fn(body, [T, O]), T)).toBe(StorageRequirement.FRESH);
  });

  it('accepts a full constant-zero initialisation, which the old heuristic rejected', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const O = new Buffer('O', [8], 'f32', 'global');
    const body = new SeqNode([
      loopOver('i', 8, (i) => new BufferStoreNode(T, [i], i32(0))),
      loopOver('j', 8, (j) => new BufferStoreNode(O, [j], new BufferLoadNode(T, [j]))),
    ]);
    expect(requirementOf(fn(body, [T, O]), T)).toBe(StorageRequirement.FRESH);
  });

  it('covers a multi-dimensional buffer only when every dim is spanned', () => {
    const T = new Buffer('T', [4, 8], 'f32', 'global');
    const O = new Buffer('O', [4, 8], 'f32', 'global');
    const full = new SeqNode([
      loopOver('i', 4, (i) => loopOver('j', 8, (j) => new BufferStoreNode(T, [i, j], i32(0)))),
      loopOver('a', 4, (a) => loopOver('b', 8, (b) => new BufferStoreNode(O, [a, b], new BufferLoadNode(T, [a, b])))),
    ]);
    expect(requirementOf(fn(full, [T, O]), T)).toBe(StorageRequirement.FRESH);

    const rowOnly = new SeqNode([
      loopOver('i', 4, (i) => new BufferStoreNode(T, [i, i32(0)], i32(0))),
      loopOver('a', 4, (a) => loopOver('b', 8, (b) => new BufferStoreNode(O, [a, b], new BufferLoadNode(T, [a, b])))),
    ]);
    expect(requirementOf(fn(rowOnly, [T, O]), T)).toBe(StorageRequirement.DEFINED);
  });
});

describe('storage requirement: unsound reuse the old heuristic allowed is now rejected', () => {
  it('rejects a partially written buffer that is then read in full', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const O = new Buffer('O', [8], 'f32', 'global');
    const A = new Buffer('A', [8], 'f32', 'global');
    const body = new SeqNode([
      new BufferStoreNode(T, [i32(0)], mathOp('*', new BufferLoadNode(A, [i32(0)]), i32(2))),
      loopOver('j', 8, (j) => new BufferStoreNode(O, [j], new BufferLoadNode(T, [j]))),
    ]);
    const fact = analyzeStorageRequirements(fn(body, [T, O, A])).get(T);
    expect(fact.requirement).toBe(StorageRequirement.DEFINED);
    expect(fact.reason).toMatch(/cover/);
  });

  it('rejects a buffer whose only write is conditional', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const O = new Buffer('O', [8], 'f32', 'global');
    const A = new Buffer('A', [8], 'f32', 'global');
    const cond = new CompareNode('lt', v('c'), i32(1));
    const body = new SeqNode([
      new IfThenElseNode(cond, loopOver('i', 8, (i) => new BufferStoreNode(T, [i], mathOp('*', new BufferLoadNode(A, [i]), i32(2))))),
      loopOver('j', 8, (j) => new BufferStoreNode(O, [j], new BufferLoadNode(T, [j]))),
    ]);
    expect(requirementOf(fn(body, [T, O, A]), T)).toBe(StorageRequirement.DEFINED);
  });

  it('rejects a buffer read before it is written', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const O = new Buffer('O', [8], 'f32', 'global');
    const body = new SeqNode([
      loopOver('j', 8, (j) => new BufferStoreNode(O, [j], new BufferLoadNode(T, [j]))),
      loopOver('i', 8, (i) => new BufferStoreNode(T, [i], i32(1))),
    ]);
    const fact = analyzeStorageRequirements(fn(body, [T, O])).get(T);
    expect(fact.requirement).toBe(StorageRequirement.DEFINED);
    expect(fact.reason).toMatch(/upward-exposed/);
  });
});

describe('storage requirement: read-modify-write stays rejected', () => {
  it('rejects scatter-add, where the store reads the buffer it writes', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const IDX = new Buffer('IDX', [8], 'i32', 'global');
    const body = new SeqNode([
      loopOver('i', 8, (i) => new BufferStoreNode(T, [i], i32(0))),
      loopOver('j', 8, (j) => {
        const slot = new BufferLoadNode(IDX, [j]);
        return new BufferStoreNode(T, [slot], mathOp('+', new BufferLoadNode(T, [slot]), i32(1)));
      }),
    ]);
    const fact = analyzeStorageRequirements(fn(body, [T, IDX])).get(T);
    expect(fact.requirement).toBe(StorageRequirement.DEFINED);
    expect(fact.reason).toMatch(/read-modify-write/);
  });

  it('rejects a store whose index reads the buffer being written', () => {
    const T = new Buffer('T', [8], 'f32', 'global');
    const body = loopOver('i', 8, (i) => new BufferStoreNode(T, [new BufferLoadNode(T, [i])], i32(1)));
    expect(requirementOf(fn(body, [T]), T)).toBe(StorageRequirement.DEFINED);
  });
});

describe('buffersRequiringDefinedStorage', () => {
  it('returns exactly the buffers that cannot sit on reused storage', () => {
    const SAFE = new Buffer('SAFE', [8], 'f32', 'global');
    const RMW = new Buffer('RMW', [8], 'f32', 'global');
    const O = new Buffer('O', [8], 'f32', 'global');
    const body = new SeqNode([
      loopOver('i', 8, (i) => new BufferStoreNode(SAFE, [i], i32(2))),
      loopOver('j', 8, (j) => new BufferStoreNode(RMW, [j], mathOp('+', new BufferLoadNode(RMW, [j]), i32(1)))),
      loopOver('k', 8, (k) => new BufferStoreNode(O, [k], mathOp('+', new BufferLoadNode(SAFE, [k]), new BufferLoadNode(RMW, [k])))),
    ]);
    const required = buffersRequiringDefinedStorage(fn(body, [SAFE, RMW, O]));
    expect(required.has(RMW)).toBe(true);
    expect(required.has(SAFE)).toBe(false);
  });
});
