import { describe, it, expect } from 'vitest';
import { scanMetadata } from '../../../../src/compiler/ir/lir/scanner.js';
import { Buffer } from '../../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode,
  BufferStoreNode, BufferLoadNode, AllocateNode,
  LetStmtNode, CallExternNode,
  MathOpNode, VariableNode, IntImmNode, FloatImmNode,
  ForKind,
} from '../../../../src/compiler/ir/tensor/nodes.js';
import { WasmTarget, CPUTarget, GPUTarget } from '../../../../src/backend/target.js';

function buf(name, shape, dtype = 'f32') {
  return new Buffer(name, shape, dtype, 'global');
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function makePrimFunc(name, params, body, bufferMap, shapeParams = []) {
  return new PrimFunc(name, params, body, bufferMap, shapeParams);
}

function simpleLoop(bufIn, bufOut) {
  const load = new BufferLoadNode(bufIn, [idx('i')]);
  const store = new BufferStoreNode(bufOut, [idx('i')], load);
  const block = new BlockNode('blk', [], [{ buffer: bufIn }], [{ buffer: bufOut }], store);
  return new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
}

describe('scanMetadata — locals', () => {
  it('collects loop variables as i32 locals', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const loop = simpleLoop(a, b);
    const pf = makePrimFunc('test', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.locals.get('i')).toBe('i32');
  });

  it('collects nested loop variables', () => {
    const m = buf('m', [3, 4]);
    const store = new BufferStoreNode(m, [idx('i'), idx('j')], new FloatImmNode(0));
    const block = new BlockNode('blk', [], [], [{ buffer: m }], store);
    const inner = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const outer = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(3), ForKind.SERIAL, inner);
    const pf = makePrimFunc('test', ['p0'], outer, new Map([['p0', m]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.locals.has('i')).toBe(true);
    expect(meta.locals.has('j')).toBe(true);
  });

  it('collects LetStmtNode variables', () => {
    const b = buf('x', [4]);
    const letVar = new VariableNode('tmp', 'f32');
    const store = new BufferStoreNode(b, [idx('i')], letVar);
    const letStmt = new LetStmtNode(letVar, new FloatImmNode(1), store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, letStmt);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.locals.has('tmp')).toBe(true);
    expect(meta.locals.get('tmp')).toBe('f32');
  });

  it('collects block iterVar bindings', () => {
    const b = buf('x', [4]);
    const store = new BufferStoreNode(b, [idx('bi')], new FloatImmNode(0));
    const block = new BlockNode('blk', [
      { iterVar: new VariableNode('bi', 'index'), binding: idx('outer') }
    ], [], [{ buffer: b }], store);
    const loop = new ForNode(idx('outer'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.locals.has('bi')).toBe(true);
  });
});

describe('scanMetadata — externCalls', () => {
  it('collects non-native extern calls', () => {
    const b = buf('x', [4]);
    const call = new CallExternNode('exp', [new BufferLoadNode(b, [idx('i')])], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], call);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.externCalls.has('exp')).toBe(true);
    expect(meta.externCalls.get('exp').argCount).toBe(1);
  });

  it('does not collect native wasm ops (sqrt, abs, etc.)', () => {
    const b = buf('x', [4]);
    const call = new CallExternNode('sqrt', [new BufferLoadNode(b, [idx('i')])], 'f32');
    const store = new BufferStoreNode(b, [idx('i')], call);
    const block = new BlockNode('blk', [], [{ buffer: b }], [{ buffer: b }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.externCalls.has('sqrt')).toBe(false);
  });

  it('collects multiple distinct extern calls', () => {
    const b = buf('x', [4]);
    const c1 = new CallExternNode('sin', [new FloatImmNode(0)], 'f32');
    const c2 = new CallExternNode('cos', [new FloatImmNode(0)], 'f32');
    const s1 = new BufferStoreNode(b, [new IntImmNode(0)], c1);
    const s2 = new BufferStoreNode(b, [new IntImmNode(1)], c2);
    const seq = new SeqNode([s1, s2]);
    const pf = makePrimFunc('test', ['p0'], seq, new Map([['p0', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.externCalls.has('sin')).toBe(true);
    expect(meta.externCalls.has('cos')).toBe(true);
  });
});

describe('scanMetadata — memoryLayout', () => {
  it('assigns offsets to buffers', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const loop = simpleLoop(a, b);
    const pf = makePrimFunc('test', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.memoryLayout.bufferOffsets.has('a')).toBe(true);
    expect(meta.memoryLayout.bufferOffsets.has('b')).toBe(true);
    expect(meta.memoryLayout.bufferOffsets.get('b')).toBeGreaterThan(0);
  });

  it('aligns buffers to 16-byte boundaries', () => {
    const a = buf('a', [1]);
    const b = buf('b', [4]);
    const loop = simpleLoop(a, b);
    const pf = makePrimFunc('test', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    const offB = meta.memoryLayout.bufferOffsets.get('b');
    expect(offB % 16).toBe(0);
  });

  it('totalBytes includes all buffers', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const loop = simpleLoop(a, b);
    const pf = makePrimFunc('test', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.memoryLayout.totalBytes).toBeGreaterThanOrEqual(32);
  });

  it('includes temp buffers found in tree', () => {
    const a = buf('a', [4]);
    const tmp = buf('tmp', [4]);
    const b = buf('b', [4]);
    const s1 = new BufferStoreNode(tmp, [idx('i')], new BufferLoadNode(a, [idx('i')]));
    const s2 = new BufferStoreNode(b, [idx('i')], new BufferLoadNode(tmp, [idx('i')]));
    const blk1 = new BlockNode('b1', [], [{ buffer: a }], [{ buffer: tmp }], s1);
    const blk2 = new BlockNode('b2', [], [{ buffer: tmp }], [{ buffer: b }], s2);
    const l1 = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk1);
    const l2 = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk2);
    const seq = new SeqNode([l1, l2]);
    const alloc = new AllocateNode(tmp, 'local', seq);
    const pf = makePrimFunc('test', ['p0', 'p1'], alloc, new Map([['p0', a], ['p1', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.memoryLayout.bufferOffsets.has('tmp')).toBe(true);
  });
});

describe('scanMetadata — zeroBuffers', () => {
  it('detects buffer written only with zeros', () => {
    const a = buf('a', [4]);
    const tmp = buf('tmp', [4]);
    const store = new BufferStoreNode(tmp, [idx('i')], new FloatImmNode(0));
    const blk = new BlockNode('blk', [], [], [{ buffer: tmp }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);
    const alloc = new AllocateNode(tmp, 'local', loop);
    const pf = makePrimFunc('test', ['p0'], alloc, new Map([['p0', a]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.zeroBuffers.has('tmp')).toBe(true);
  });

  it('does not mark param buffers as zero', () => {
    const a = buf('a', [4]);
    const store = new BufferStoreNode(a, [idx('i')], new FloatImmNode(0));
    const blk = new BlockNode('blk', [], [], [{ buffer: a }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', a]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.zeroBuffers.has('a')).toBe(false);
  });

  it('detects constant buffers', () => {
    const tmp = buf('tmp', [4]);
    const store = new BufferStoreNode(tmp, [idx('i')], new FloatImmNode(42));
    const blk = new BlockNode('blk', [], [], [{ buffer: tmp }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);
    const alloc = new AllocateNode(tmp, 'local', loop);
    const pf = makePrimFunc('test', [], alloc, new Map());
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.constantBuffers.get('tmp')).toBe(42);
  });
});

describe('scanMetadata — threadBindings (GPU)', () => {
  it('collects thread bindings from ForNode', () => {
    const b = buf('x', [256]);
    const store = new BufferStoreNode(b, [idx('tid')], new FloatImmNode(0));
    const blk = new BlockNode('blk', [], [], [{ buffer: b }], store);
    const loop = new ForNode(idx('tid'), new IntImmNode(0), new IntImmNode(256), ForKind.THREAD_BINDING, blk);
    loop.threadTag = 'threadIdx.x';
    const pf = makePrimFunc('test', ['p0'], loop, new Map([['p0', b]]));
    const meta = scanMetadata(pf, GPUTarget());
    expect(meta.threadBindings.has('threadIdx.x')).toBe(true);
    expect(meta.threadBindings.get('threadIdx.x')[0].varName).toBe('tid');
    expect(meta.threadBindings.get('threadIdx.x')[0].extent).toBe(256);
  });
});

describe('scanMetadata — sharedBuffers', () => {
  it('collects shared-scope allocations', () => {
    const shared = buf('smem', [64]);
    const store = new BufferStoreNode(shared, [idx('i')], new FloatImmNode(0));
    const blk = new BlockNode('blk', [], [], [{ buffer: shared }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(64), ForKind.SERIAL, blk);
    const alloc = new AllocateNode(shared, 'shared', loop);
    const pf = makePrimFunc('test', [], alloc, new Map());
    const meta = scanMetadata(pf, GPUTarget());
    expect(meta.sharedBuffers.length).toBe(1);
    expect(meta.sharedBuffers[0].name).toBe('smem');
  });
});

describe('scanMetadata — paramBuffers', () => {
  it('identifies param buffers', () => {
    const a = buf('a', [4]);
    const b = buf('b', [4]);
    const loop = simpleLoop(a, b);
    const pf = makePrimFunc('test', ['p0', 'p1'], loop, new Map([['p0', a], ['p1', b]]));
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.paramBuffers.has('a')).toBe(true);
    expect(meta.paramBuffers.has('b')).toBe(true);
  });
});

describe('scanMetadata — allocatedBuffers', () => {
  it('tracks allocated buffer names', () => {
    const tmp = buf('tmp', [4]);
    const store = new BufferStoreNode(tmp, [idx('i')], new FloatImmNode(0));
    const blk = new BlockNode('blk', [], [], [{ buffer: tmp }], store);
    const loop = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, blk);
    const alloc = new AllocateNode(tmp, 'local', loop);
    const pf = makePrimFunc('test', [], alloc, new Map());
    const meta = scanMetadata(pf, WasmTarget());
    expect(meta.allocatedBuffers.has('tmp')).toBe(true);
  });
});
