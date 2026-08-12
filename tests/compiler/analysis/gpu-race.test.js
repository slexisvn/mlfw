import { describe, it, expect } from 'vitest';
import {
  PrimFunc, ForNode, ForKind, SeqNode, AllocateNode,
  BufferStoreNode, BufferLoadNode, VariableNode, IntImmNode, MathOpNode, FloatImmNode,
} from '../../../src/compiler/ir/tensor/nodes.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  profileGpuAccesses,
  crossBlockRAWBuffers,
  threadSharedIntermediates,
  loopCarriedIntermediates,
  extentMismatchBuffers,
  storedUnderBlockBinding,
  hasMultiExtentBlockBinding,
} from '../../../src/compiler/analysis/gpu_race.js';

const v = (name) => new VariableNode(name, 'int32');
const i32 = (n) => new IntImmNode(n);
const f32 = (n) => new FloatImmNode(n);
const add = (a, b) => new MathOpNode('+', a, b);

const bound = (tag, extent, body) => new ForNode(v(tag.replace('.', '_')), i32(0), i32(extent), ForKind.THREAD_BINDING, body, tag);
const serial = (name, extent, body) => new ForNode(v(name), i32(0), i32(extent), ForKind.SERIAL, body);

const global = (name, shape) => new Buffer(name, shape, 'float32', 'global');
const local = (name, shape) => new Buffer(name, shape, 'float32', 'local');

function func(body, storage) {
  const map = new Map();
  for (const buf of storage) map.set(v(buf.name), buf);
  return new PrimFunc('k', [], body, map);
}

function bindings(entries) {
  const m = new Map();
  for (const [tag, extents] of entries) {
    m.set(tag, extents.map((extent, i) => ({ varName: `${tag}_${i}`, extent, isDynamic: false, extentNode: i32(extent) })));
  }
  return m;
}

describe('profileGpuAccesses', () => {
  it('records a storage buffer written and read under one binding signature as race-free', () => {
    const A = global('A', [64]);
    const body = bound('blockIdx.x', 8, bound('threadIdx.x', 8, new SeqNode([
      new BufferStoreNode(A, [v('blockIdx_x')], f32(1)),
      new BufferStoreNode(A, [v('blockIdx_x')], add(new BufferLoadNode(A, [v('blockIdx_x')]), f32(1))),
    ])));

    const profile = profileGpuAccesses(func(body, [A]));
    expect(crossBlockRAWBuffers(profile).size).toBe(0);
  });

  it('flags a storage buffer written under one launch geometry and read under another', () => {
    const A = global('A', [64]);
    const B = global('B', [64]);
    const produce = bound('blockIdx.x', 8, bound('threadIdx.x', 8, new BufferStoreNode(A, [v('blockIdx_x')], f32(1))));
    const consume = bound('blockIdx.x', 4, new BufferStoreNode(B, [v('blockIdx_x')], new BufferLoadNode(A, [v('blockIdx_x')])));

    const profile = profileGpuAccesses(func(new SeqNode([produce, consume]), [A, B]));
    expect([...crossBlockRAWBuffers(profile)]).toEqual(['A']);
  });

  it('does not flag a storage buffer that is only written', () => {
    const A = global('A', [64]);
    const body = new SeqNode([
      bound('blockIdx.x', 8, new BufferStoreNode(A, [v('blockIdx_x')], f32(0))),
      bound('blockIdx.x', 4, new BufferStoreNode(A, [v('blockIdx_x')], f32(1))),
    ]);
    const profile = profileGpuAccesses(func(body, [A]));
    expect(crossBlockRAWBuffers(profile).size).toBe(0);
  });
});

describe('threadSharedIntermediates', () => {
  it('flags a multi-element kernel-local buffer written and read by different threads', () => {
    const out = global('out', [8]);
    const tmp = local('tmp', [8]);
    const body = new AllocateNode(tmp, 'local', bound('threadIdx.x', 8, new SeqNode([
      new BufferStoreNode(tmp, [v('threadIdx_x')], f32(1)),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(tmp, [i32(0)])),
    ])));

    const profile = profileGpuAccesses(func(body, [out]), { threadBindings: bindings([['threadIdx.x', [8]]]) });
    expect([...threadSharedIntermediates(profile)]).toEqual(['tmp']);
  });

  it('excludes a buffer allocated inside the thread binding as thread-private', () => {
    const out = global('out', [8]);
    const tmp = local('tmp', [8]);
    const body = bound('threadIdx.x', 8, new AllocateNode(tmp, 'local', new SeqNode([
      new BufferStoreNode(tmp, [i32(0)], f32(1)),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(tmp, [i32(0)])),
    ])));

    const profile = profileGpuAccesses(func(body, [out]), { threadBindings: bindings([['threadIdx.x', [8]]]) });
    expect(threadSharedIntermediates(profile).size).toBe(0);
  });

  it('excludes buffers already living in shared memory', () => {
    const out = global('out', [8]);
    const tmp = new Buffer('tmp', [8], 'float32', 'shared');
    const body = new AllocateNode(tmp, 'shared', bound('threadIdx.x', 8, new SeqNode([
      new BufferStoreNode(tmp, [v('threadIdx_x')], f32(1)),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(tmp, [i32(0)])),
    ])));

    const profile = profileGpuAccesses(func(body, [out]), { sharedBuffers: [tmp], threadBindings: bindings([['threadIdx.x', [8]]]) });
    expect(threadSharedIntermediates(profile).size).toBe(0);
  });

  it('flags a scalar intermediate only when it is written under a narrowed binding', () => {
    const out = global('out', [8]);
    const acc = local('acc', [1]);
    const narrowed = new AllocateNode(acc, 'local', bound('threadIdx.x', 8, new SeqNode([
      bound('threadIdx.x', 1, new BufferStoreNode(acc, [i32(0)], f32(0))),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(acc, [i32(0)])),
    ])));
    const wide = new AllocateNode(acc, 'local', bound('threadIdx.x', 8, new SeqNode([
      new BufferStoreNode(acc, [i32(0)], f32(0)),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(acc, [i32(0)])),
    ])));
    const tb = bindings([['threadIdx.x', [8, 1]]]);

    expect([...threadSharedIntermediates(profileGpuAccesses(func(narrowed, [out]), { threadBindings: tb }))]).toEqual(['acc']);
    expect(threadSharedIntermediates(profileGpuAccesses(func(wide, [out]), { threadBindings: tb })).size).toBe(0);
  });
});

describe('storedUnderBlockBinding', () => {
  it('separates block-local intermediates from grid-wide ones', () => {
    const out = global('out', [8]);
    const tmp = local('tmp', [8]);
    const blockLocal = new AllocateNode(tmp, 'local', bound('threadIdx.x', 8, new SeqNode([
      new BufferStoreNode(tmp, [v('threadIdx_x')], f32(1)),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(tmp, [i32(0)])),
    ])));
    const gridWide = new AllocateNode(tmp, 'local', bound('blockIdx.x', 4, bound('threadIdx.x', 8, new SeqNode([
      new BufferStoreNode(tmp, [v('threadIdx_x')], f32(1)),
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(tmp, [i32(0)])),
    ]))));

    const pLocal = profileGpuAccesses(func(blockLocal, [out]));
    const pGrid = profileGpuAccesses(func(gridWide, [out]));
    expect(storedUnderBlockBinding(pLocal, threadSharedIntermediates(pLocal))).toBe(false);
    expect(storedUnderBlockBinding(pGrid, threadSharedIntermediates(pGrid))).toBe(true);
  });
});

describe('loopCarriedIntermediates', () => {
  it('flags an intermediate whose load index depends on an enclosing sequential loop', () => {
    const out = global('out', [8]);
    const state = local('state', [8]);
    const body = new AllocateNode(state, 'local', bound('threadIdx.x', 8, serial('t', 4,
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(state, [v('t')])))));

    const profile = profileGpuAccesses(func(body, [out]));
    expect([...loopCarriedIntermediates(profile)]).toEqual(['state']);
  });

  it('does not flag an intermediate indexed only by thread ids', () => {
    const out = global('out', [8]);
    const state = local('state', [8]);
    const body = new AllocateNode(state, 'local', bound('threadIdx.x', 8, serial('t', 4,
      new BufferStoreNode(out, [v('threadIdx_x')], new BufferLoadNode(state, [v('threadIdx_x')])))));

    const profile = profileGpuAccesses(func(body, [out]));
    expect(loopCarriedIntermediates(profile).size).toBe(0);
  });
});

describe('extentMismatchBuffers', () => {
  it('flags an intermediate written by many threads and read by one', () => {
    const out = global('out', [8]);
    const tmp = local('tmp', [8]);
    const body = new AllocateNode(tmp, 'local', new SeqNode([
      bound('threadIdx.x', 8, new BufferStoreNode(tmp, [v('threadIdx_x')], add(f32(1), f32(2)))),
      new BufferStoreNode(out, [i32(0)], new BufferLoadNode(tmp, [i32(0)])),
    ]));

    const profile = profileGpuAccesses(func(body, [out]));
    expect([...extentMismatchBuffers(profile)]).toEqual(['tmp']);
  });

  it('ignores an intermediate whose stores are all literals', () => {
    const out = global('out', [8]);
    const tmp = local('tmp', [8]);
    const body = new AllocateNode(tmp, 'local', new SeqNode([
      bound('threadIdx.x', 8, new BufferStoreNode(tmp, [v('threadIdx_x')], f32(0))),
      new BufferStoreNode(out, [i32(0)], new BufferLoadNode(tmp, [i32(0)])),
    ]));

    const profile = profileGpuAccesses(func(body, [out]));
    expect(extentMismatchBuffers(profile).size).toBe(0);
  });
});

describe('hasMultiExtentBlockBinding', () => {
  it('distinguishes a block axis bound at two extents from a thread axis', () => {
    expect(hasMultiExtentBlockBinding(bindings([['blockIdx.x', [4, 8]]]))).toEqual({ blockSpace: true, threadSpace: false });
    expect(hasMultiExtentBlockBinding(bindings([['threadIdx.x', [4, 8]]]))).toEqual({ blockSpace: false, threadSpace: true });
    expect(hasMultiExtentBlockBinding(bindings([['blockIdx.x', [8, 8]]]))).toEqual({ blockSpace: false, threadSpace: false });
  });
});
