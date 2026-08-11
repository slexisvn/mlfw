import { describe, it, expect } from 'vitest';
import { BackendPipeline, CompiledKernel } from '../../src/backend/pipeline.js';
import { CUDATarget } from '../../src/backend/target.js';
import { Buffer } from '../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode,
  BufferStoreNode, BufferLoadNode, AllocateNode,
  MathOpNode, CallExternNode,
  VariableNode, IntImmNode, FloatImmNode,
  ForKind
} from '../../src/compiler/ir/tensor/nodes.js';

function buf(name, shape, dtype = 'f32', scope = 'global') {
  return new Buffer(name, shape, dtype, scope);
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function makePrimFunc(name, params, body, bufferMap, shapeParams = []) {
  return new PrimFunc(name, params, body, bufferMap, shapeParams);
}

function threadFor(loopVar, extent, tag, body) {
  return new ForNode(loopVar, new IntImmNode(0), new IntImmNode(extent), ForKind.THREAD_BINDING, body, tag);
}

describe('BackendPipeline — GPU compile', () => {
  it('returns CompiledKernel with cuda metadata', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const inBuf = buf('x', [256], 'f32');
    const outBuf = buf('y', [256], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const store = new BufferStoreNode(outBuf, [idx('tid')], load);
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy_gpu', ['p0', 'p1'], body, bufferMap);

    const result = pipeline.compile(pf);

    expect(result).toBeInstanceOf(CompiledKernel);
    expect(result.name).toBe('copy_gpu');
    expect(result.metadata.kind).toBe('cuda');
    expect(result.metadata.blockDim).toEqual([256, 1, 1]);
    expect(result.metadata.gridDim).toBeDefined();
    expect(result.metadata.params).toContain('x');
    expect(result.metadata.params).toContain('y');
    expect(result.source).toMatch(/__global__ void copy_gpu/);
  });

  it('preserves shared memory info in metadata', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const smem = buf('smem', [64], 'f32', 'shared');
    const outBuf = buf('out', [64], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const alloc = new AllocateNode(smem, 'shared', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('smem_kernel', ['p0'], alloc, bufferMap);

    const result = pipeline.compile(pf);
    expect(result.metadata.sharedMemBytes).toBe(64 * 4);
  });
});

describe('BackendPipeline.compileAll', () => {
  it('compiles multiple PrimFuncs', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const funcs = [1, 2, 3].map(i => {
      const inBuf = buf(`x${i}`, [64], 'f32');
      const outBuf = buf(`y${i}`, [64], 'f32');
      const store = new BufferStoreNode(outBuf, [idx('j')], new BufferLoadNode(inBuf, [idx('j')]));
      const forNode = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(64), ForKind.SERIAL, store);
      const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
      return makePrimFunc(`kernel_${i}`, ['p0', 'p1'], forNode, bufferMap);
    });

    const results = pipeline.compileAll(funcs);
    expect(results).toHaveLength(3);
    results.forEach((r, i) => {
      expect(r.name).toBe(`kernel_${i + 1}`);
      expect(r.metadata.kind).toBe('cuda');
    });
  });
});

describe('BackendPipeline — GPU elementwise kernel correctness', () => {
  it('generates valid CUDA for vector negate', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const neg = new MathOpNode('-', load);
    const store = new BufferStoreNode(outBuf, [idx('i')], neg);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('negate_gpu', ['p0', 'p1'], forNode, bufferMap);

    const result = pipeline.compile(pf);
    expect(result.source).toMatch(/y\[i\] = \(-x\[i\]\)/);
  });

  it('generates valid CUDA for fused add+exp', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const aBuf = buf('a', [4], 'f32');
    const bBuf = buf('b', [4], 'f32');
    const outBuf = buf('c', [4], 'f32');

    const loadA = new BufferLoadNode(aBuf, [idx('i')]);
    const loadB = new BufferLoadNode(bBuf, [idx('i')]);
    const add = new MathOpNode('+', loadA, loadB);
    const expCall = new CallExternNode('exp', [add], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], expCall);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', aBuf], ['p1', bBuf], ['p2', outBuf]]);
    const pf = makePrimFunc('add_exp_gpu', ['p0', 'p1', 'p2'], forNode, bufferMap);

    const result = pipeline.compile(pf);
    expect(result.source).toMatch(/expf\(\(a\[i\] \+ b\[i\]\)\)/);
  });
});

describe('BackendPipeline — GPU with thread-bound kernels', () => {
  it('produces correct block/grid dimensions in metadata', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const outBuf = buf('out', [1024], 'f32');
    const store = new BufferStoreNode(outBuf, [
      new MathOpNode('+', new MathOpNode('*', idx('bid'), new IntImmNode(128)), idx('tid'))
    ], new FloatImmNode(0));

    const inner = threadFor(idx('tid'), 128, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 8, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('fill_gpu', ['p0'], outer, bufferMap);

    const result = pipeline.compile(pf);
    expect(result.metadata.blockDim).toEqual([128, 1, 1]);
    expect(result.metadata.gridDim).toEqual([8, 1, 1]);
    expect(result.metadata.sharedMemBytes).toBe(0);
  });
});

describe('BackendPipeline — GPU kernel with mixed loops', () => {
  it('generates serial for loop inside thread binding', () => {
    const pipeline = new BackendPipeline(CUDATarget());

    const inBuf = buf('inp', [32, 16], 'f32');
    const outBuf = buf('out', [32], 'f32');

    const initStore = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));
    const accLoad = new BufferLoadNode(outBuf, [idx('tid')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('tid'), idx('k')]);
    const add = new MathOpNode('+', accLoad, inLoad);
    const accStore = new BufferStoreNode(outBuf, [idx('tid')], add);
    const innerFor = new ForNode(idx('k'), new IntImmNode(0), new IntImmNode(16), ForKind.SERIAL, accStore);
    const block = new BlockNode('reduce_blk', [], [{ buffer: inBuf }], [{ buffer: outBuf }], innerFor, initStore);

    const bindTid = threadFor(idx('tid'), 32, 'threadIdx.x', block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('reduce_gpu', ['p0', 'p1'], bindTid, bufferMap);

    const result = pipeline.compile(pf);
    expect(result.source).toMatch(/for \(int k = 0; k < 16; k\+\+\)/);
    expect(result.source).toMatch(/const int tid = threadIdx\.x/);
    expect(result.metadata.blockDim[0]).toBe(32);
  });
});
