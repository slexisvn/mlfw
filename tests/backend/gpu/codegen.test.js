import { describe, it, expect } from 'vitest';
import { GPUCodegen, GPUKernel } from '../../../src/backend/gpu/codegen.js';
import { GPUTarget } from '../../../src/backend/target.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode,
  BufferStoreNode, BufferLoadNode, AllocateNode,
  MathOpNode, CompareNode, IfThenElseNode, CastNode,
  LetStmtNode, CallExternNode, WhileNode,
  VariableNode, IntImmNode, FloatImmNode,
  ForKind
} from '../../../src/compiler/ir/tensor/nodes.js';

function makeCodegen(overrides = {}) {
  return new GPUCodegen(GPUTarget(overrides));
}

function buf(name, shape, dtype = 'f32', scope = 'global') {
  return new Buffer(name, shape, dtype, scope);
}

function idx(name) {
  return new VariableNode(name, 'index');
}

function makePrimFunc(name, params, body, bufferMap, shapeParams = [], shapeParamMap = null) {
  return new PrimFunc(name, params, body, bufferMap, shapeParams, shapeParamMap);
}

function threadFor(loopVar, extent, tag, body) {
  return new ForNode(loopVar, new IntImmNode(0), new IntImmNode(extent), ForKind.THREAD_BINDING, body, tag);
}

describe('GPUCodegen._exprToC', () => {
  function exprToC(node) {
    const cg = makeCodegen();
    cg._defaultDtype = 'f32';
    return cg._exprToC(node);
  }

  it('renders IntImmNode', () => {
    expect(exprToC(new IntImmNode(42))).toBe('42');
    expect(exprToC(new IntImmNode(0))).toBe('0');
    expect(exprToC(new IntImmNode(-7))).toBe('-7');
  });

  it('renders FloatImmNode with suffix', () => {
    const cg = makeCodegen();
    cg._defaultDtype = 'f32';
    expect(cg._exprToC(new FloatImmNode(3.14))).toBe('3.14f');
    cg._defaultDtype = 'f64';
    expect(cg._exprToC(new FloatImmNode(3.14))).toBe('3.14');
  });

  it('renders Infinity as INFINITY', () => {
    const cg = makeCodegen();
    cg._defaultDtype = 'f32';
    expect(cg._exprToC(new FloatImmNode(Infinity))).toBe('INFINITY');
    expect(cg._exprToC(new FloatImmNode(-Infinity))).toBe('(-INFINITY)');
  });

  it('renders VariableNode', () => {
    expect(exprToC(new VariableNode('threadIdx_x', 'index'))).toBe('threadIdx_x');
  });

  it('renders binary MathOpNode', () => {
    const add = new MathOpNode('+', new VariableNode('a', 'f32'), new VariableNode('b', 'f32'));
    expect(exprToC(add)).toBe('(a + b)');

    const mul = new MathOpNode('*', new VariableNode('i', 'index'), new IntImmNode(8));
    expect(exprToC(mul)).toBe('(i * 8)');
  });

  it('renders unary MathOpNode (neg)', () => {
    const neg = new MathOpNode('-', new VariableNode('x', 'f32'));
    expect(exprToC(neg)).toBe('(-x)');
  });

  it('renders integer division as C division', () => {
    const div = new MathOpNode('//', new VariableNode('x', 'index'), new IntImmNode(4));
    expect(exprToC(div)).toBe('(x / 4)');
  });

  it('renders nested expressions', () => {
    const inner = new MathOpNode('+', new VariableNode('i', 'index'), new IntImmNode(1));
    const outer = new MathOpNode('*', inner, new IntImmNode(16));
    expect(exprToC(outer)).toBe('((i + 1) * 16)');
  });

  it('renders CompareNode with C operators', () => {
    const ge = new CompareNode('ge', new VariableNode('x', 'index'), new IntImmNode(0));
    expect(exprToC(ge)).toBe('(x >= 0)');

    const eq = new CompareNode('eq', new VariableNode('a', 'index'), new VariableNode('b', 'index'));
    expect(exprToC(eq)).toBe('(a == b)');

    const ne = new CompareNode('ne', new VariableNode('a', 'index'), new IntImmNode(0));
    expect(exprToC(ne)).toBe('(a != 0)');
  });

  it('renders IfThenElseNode as ternary', () => {
    const cond = new CompareNode('gt', new VariableNode('x', 'f32'), new FloatImmNode(0));
    const node = new IfThenElseNode(cond, new FloatImmNode(1), new FloatImmNode(0));
    expect(exprToC(node)).toBe('((x > 0f) ? 1f : 0f)');
  });

  it('renders CastNode', () => {
    const node = new CastNode(new FloatImmNode(3.7), 'f32', 'i32');
    expect(exprToC(node)).toBe('((int)(3.7f))');
  });

  it('renders BufferLoadNode with flat index', () => {
    const b = buf('data', [4], 'f32');
    const node = new BufferLoadNode(b, [new VariableNode('i', 'index')]);
    expect(exprToC(node)).toBe('data[i]');
  });

  it('renders BufferLoadNode with multi-dim index', () => {
    const b = buf('mat', [3, 4], 'f32');
    const node = new BufferLoadNode(b, [idx('i'), idx('j')]);
    expect(exprToC(node)).toBe('mat[i * 4 + j]');
  });

  it('returns 0 for null input', () => {
    expect(exprToC(null)).toBe('0');
  });

  it('returns 0 for unknown node types', () => {
    expect(exprToC({ type: 'UnknownNode' })).toBe('0');
  });
});

describe('GPUCodegen._emitExternCall', () => {
  function externCall(name, args, dtype = 'f32') {
    const cg = makeCodegen();
    cg._defaultDtype = dtype;
    return cg._emitExternCall(new CallExternNode(name, args, dtype));
  }

  it('maps math functions with f32 suffix', () => {
    const x = new VariableNode('x', 'f32');
    expect(externCall('exp', [x])).toBe('expf(x)');
    expect(externCall('log', [x])).toBe('logf(x)');
    expect(externCall('sqrt', [x])).toBe('sqrtf(x)');
    expect(externCall('tanh', [x])).toBe('tanhf(x)');
    expect(externCall('sin', [x])).toBe('sinf(x)');
    expect(externCall('cos', [x])).toBe('cosf(x)');
    expect(externCall('abs', [x])).toBe('fabsf(x)');
    expect(externCall('ceil', [x])).toBe('ceilf(x)');
    expect(externCall('floor', [x])).toBe('floorf(x)');
  });

  it('maps math functions with f64 (no suffix)', () => {
    const x = new VariableNode('x', 'f64');
    expect(externCall('exp', [x], 'f64')).toBe('exp(x)');
    expect(externCall('log', [x], 'f64')).toBe('log(x)');
    expect(externCall('sqrt', [x], 'f64')).toBe('sqrt(x)');
    expect(externCall('abs', [x], 'f64')).toBe('fabs(x)');
  });

  it('maps rsqrt with fallback', () => {
    const x = new VariableNode('x', 'f32');
    const result = externCall('rsqrt', [x]);
    expect(result).toMatch(/rsqrt.*\(x\)/);
  });

  it('emits sign as comparison expression', () => {
    const x = new VariableNode('x', 'f32');
    const result = externCall('sign', [x]);
    expect(result).toContain('(x > 0.0f)');
    expect(result).toContain('(x < 0.0f)');
    expect(result).toMatch(/\(x > 0\.0f\) - \(x < 0\.0f\)/);
  });

  it('handles two-argument functions (max, min, pow)', () => {
    const a = new VariableNode('a', 'f32');
    const b = new VariableNode('b', 'f32');
    expect(externCall('max', [a, b])).toBe('fmaxf(a, b)');
    expect(externCall('min', [a, b])).toBe('fminf(a, b)');
    expect(externCall('pow', [a, b])).toBe('powf(a, b)');
  });
});

describe('GPUCodegen._flatIndex', () => {
  function flatIndex(buffer, indices) {
    const cg = makeCodegen();
    cg._primFunc = null;
    return cg._flatIndex(buffer, indices);
  }

  it('scalar (no indices) returns 0', () => {
    expect(flatIndex(buf('s', []), [])).toBe('0');
  });

  it('1D returns the index directly', () => {
    expect(flatIndex(buf('v', [8]), [idx('i')])).toBe('i');
  });

  it('2D computes row-major flat index', () => {
    const result = flatIndex(buf('m', [3, 4]), [idx('i'), idx('j')]);
    expect(result).toBe('i * 4 + j');
  });

  it('3D computes correct strides', () => {
    const result = flatIndex(buf('t', [2, 3, 4]), [idx('i'), idx('j'), idx('k')]);
    expect(result).toBe('i * 12 + j * 4 + k');
  });

  it('4D computes correct strides for batch tensor', () => {
    const result = flatIndex(buf('t', [2, 3, 4, 5]), [idx('b'), idx('c'), idx('h'), idx('w')]);
    expect(result).toBe('b * 60 + c * 20 + h * 5 + w');
  });

  it('handles stride-1 dimension without multiplication', () => {
    const b = buf('v', [4, 1]);
    const result = flatIndex(b, [idx('i'), idx('j')]);
    expect(result).toBe('i + j');
  });
});

describe('GPUCodegen._scanBindings', () => {
  it('extracts threadIdx.x binding and sets blockDim', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [256]), [idx('tid')], new FloatImmNode(0));
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    cg._scanBindings(body);
    expect(cg._threadBindings.has('threadIdx.x')).toBe(true);
    expect(cg._threadBindings.get('threadIdx.x')[0].varName).toBe('tid');
    expect(cg._blockDim[0]).toBe(256);
  });

  it('extracts blockIdx.x binding and sets gridDim', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [1024]), [idx('bid')], new FloatImmNode(0));
    const body = threadFor(idx('bid'), 64, 'blockIdx.x', store);
    cg._scanBindings(body);
    expect(cg._threadBindings.has('blockIdx.x')).toBe(true);
    expect(cg._gridDim[0]).toBe(64);
  });

  it('extracts threadIdx.y and sets blockDim[1]', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [16]), [idx('ty')], new FloatImmNode(0));
    const body = threadFor(idx('ty'), 16, 'threadIdx.y', store);
    cg._scanBindings(body);
    expect(cg._blockDim[1]).toBe(16);
  });

  it('extracts blockIdx.y and sets gridDim[1]', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [32]), [idx('by')], new FloatImmNode(0));
    const body = threadFor(idx('by'), 32, 'blockIdx.y', store);
    cg._scanBindings(body);
    expect(cg._gridDim[1]).toBe(32);
  });

  it('finds nested thread bindings', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [1024]), [idx('tid')], new FloatImmNode(0));
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);
    cg._scanBindings(outer);
    expect(cg._blockDim[0]).toBe(256);
    expect(cg._gridDim[0]).toBe(4);
  });

  it('finds shared memory allocations', () => {
    const cg = makeCodegen();
    const sharedBuf = buf('smem', [128], 'f32', 'shared');
    const store = new BufferStoreNode(buf('out', [128]), [idx('i')], new FloatImmNode(0));
    const alloc = new AllocateNode(sharedBuf, 'shared', store);
    cg._scanBindings(alloc);
    expect(cg._sharedBuffers).toHaveLength(1);
    expect(cg._sharedBuffers[0].name).toBe('smem');
  });

  it('handles dynamic extent (non-IntImmNode)', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [256]), [idx('tid')], new FloatImmNode(0));
    const dynExtent = new VariableNode('N', 'index');
    const body = new ForNode(idx('tid'), new IntImmNode(0), dynExtent, ForKind.THREAD_BINDING, store, 'threadIdx.x');
    cg._scanBindings(body);
    expect(cg._threadBindings.get('threadIdx.x')[0].isDynamic).toBe(true);
    expect(cg._blockDim[0]).toBe(1);
  });
});

describe('GPUCodegen._applyBindingDim', () => {
  it('maps threadIdx.x to blockDim[0]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.x', 128);
    expect(cg._blockDim).toEqual([128, 1, 1]);
  });

  it('maps threadIdx.y to blockDim[1]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.y', 16);
    expect(cg._blockDim).toEqual([1, 16, 1]);
  });

  it('maps threadIdx.z to blockDim[2]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.z', 4);
    expect(cg._blockDim).toEqual([1, 1, 4]);
  });

  it('maps blockIdx.x to gridDim[0]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('blockIdx.x', 32);
    expect(cg._gridDim).toEqual([32, 1, 1]);
  });

  it('maps blockIdx.y to gridDim[1]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('blockIdx.y', 64);
    expect(cg._gridDim).toEqual([1, 64, 1]);
  });

  it('ignores tags without dot separator', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('noprefix', 100);
    expect(cg._blockDim).toEqual([1, 1, 1]);
    expect(cg._gridDim).toEqual([1, 1, 1]);
  });

  it('ignores invalid axis character', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.w', 100);
    expect(cg._blockDim).toEqual([1, 1, 1]);
  });
});

describe('GPUCodegen.generate — kernel signature', () => {
  it('emits __global__ void with correct param types', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy_kernel', ['p0', 'p1'], forNode, bufferMap);

    const cg = makeCodegen();
    const kernel = cg.generate(pf);
    expect(kernel.source).toMatch(/__global__ void copy_kernel\(float\* x, float\* y\)/);
    expect(kernel.name).toBe('copy_kernel');
    expect(kernel.params).toEqual(['x', 'y']);
  });

  it('emits correct pointer types for different dtypes', () => {
    const inBuf = buf('x', [4], 'f64');
    const outBuf = buf('y', [4], 'f64');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy_f64', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/double\* x, double\* y/);
  });

  it('includes shape params as int args', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const nVar = new VariableNode('N', 'index');
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('dynamic_kernel', ['p0', 'p1'], forNode, bufferMap, [nVar]);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/int N/);
    expect(kernel.params).toContain('N');
  });
});

describe('GPUCodegen.generate — thread bindings', () => {
  it('emits const int assignments for thread bindings', () => {
    const inBuf = buf('x', [1024], 'f32');
    const outBuf = buf('y', [1024], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const store = new BufferStoreNode(outBuf, [idx('tid')], load);
    const block = new BlockNode('cp', [], [{ buffer: inBuf }], [{ buffer: outBuf }], store);
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', block);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('thread_test', ['p0', 'p1'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/const int tid = threadIdx\.x;/);
    expect(kernel.source).toMatch(/const int bid = blockIdx\.x;/);
    expect(kernel.blockDim).toEqual([256, 1, 1]);
    expect(kernel.gridDim).toEqual([4, 1, 1]);
  });

  it('sets 2D block/grid dimensions', () => {
    const outBuf = buf('out', [16, 16], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('ty'), idx('tx')], new FloatImmNode(0));
    const block = new BlockNode('fill', [], [], [{ buffer: outBuf }], store);
    const bindTx = threadFor(idx('tx'), 16, 'threadIdx.x', block);
    const bindTy = threadFor(idx('ty'), 16, 'threadIdx.y', bindTx);
    const bindBx = threadFor(idx('bx'), 4, 'blockIdx.x', bindTy);
    const bindBy = threadFor(idx('by'), 4, 'blockIdx.y', bindBx);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('grid2d', ['p0'], bindBy, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.blockDim).toEqual([16, 16, 1]);
    expect(kernel.gridDim).toEqual([4, 4, 1]);
  });

  it('emits const int assignment instead of for loop for thread binding', () => {
    const outBuf = buf('out', [256], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(1));
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('binding_test', ['p0'], body, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/const int tid = threadIdx\.x/);
    expect(kernel.source).toMatch(/out\[tid\] = 1f/);
  });
});

describe('GPUCodegen.generate — for loops', () => {
  it('emits serial for loop with C syntax', () => {
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('serial_loop', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/for \(int i = 0; i < 4; i\+\+\)/);
  });

  it('emits #pragma unroll for UNROLLED loops', () => {
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.UNROLLED, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('unroll_loop', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/#pragma unroll/);
    expect(kernel.source).toMatch(/for \(int i = 0; i < 4; i\+\+\)/);
  });
});

describe('GPUCodegen.generate — shared memory', () => {
  it('declares __shared__ buffers and computes sharedMemBytes', () => {
    const sharedBuf = buf('smem', [64], 'f32', 'shared');
    const outBuf = buf('out', [64], 'f32');

    const load = new BufferLoadNode(sharedBuf, [idx('i')]);
    const store = new BufferStoreNode(outBuf, [idx('i')], load);
    const alloc = new AllocateNode(sharedBuf, 'shared', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('smem_test', ['p0'], alloc, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/__shared__ float smem\[64\]/);
    expect(kernel.sharedMemBytes).toBe(64 * 4);
  });

  it('accumulates shared memory from multiple buffers', () => {
    const sm1 = buf('smem_a', [32], 'f32', 'shared');
    const sm2 = buf('smem_b', [16], 'f64', 'shared');
    const outBuf = buf('out', [32], 'f32');

    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const inner = new AllocateNode(sm2, 'shared', store);
    const outer = new AllocateNode(sm1, 'shared', inner);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('multi_smem', ['p0'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.sharedMemBytes).toBe(32 * 4 + 16 * 8);
    expect(kernel.source).toMatch(/__shared__ float smem_a\[32\]/);
    expect(kernel.source).toMatch(/__shared__ double smem_b\[16\]/);
  });
});

describe('GPUCodegen.generate — local allocation', () => {
  it('emits local array declaration for non-shared AllocateNode', () => {
    const localBuf = buf('local_buf', [8], 'f32', 'local');
    const outBuf = buf('out', [8], 'f32');

    const store = new BufferStoreNode(localBuf, [idx('i')], new FloatImmNode(0));
    const load = new BufferLoadNode(localBuf, [idx('i')]);
    const storeOut = new BufferStoreNode(outBuf, [idx('i')], load);
    const seq = new SeqNode([store, storeOut]);
    const alloc = new AllocateNode(localBuf, 'local', seq);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('local_alloc', ['p0'], alloc, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/float local_buf\[8\]/);
  });
});

describe('GPUCodegen.generate — if/else statements', () => {
  it('emits if block for conditional store', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const cond = new CompareNode('gt', new BufferLoadNode(inBuf, [idx('i')]), new FloatImmNode(0));
    const storeThen = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const storeElse = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const ifStmt = new IfThenElseNode(cond, storeThen, storeElse);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, ifStmt);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('relu_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/if\s*\(/);
    expect(kernel.source).toMatch(/\} else \{/);
  });

  it('emits if without else when elseBody is null', () => {
    const outBuf = buf('y', [4], 'f32');
    const cond = new CompareNode('lt', idx('i'), new IntImmNode(3));
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(1));
    const ifStmt = new IfThenElseNode(cond, store, null);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, ifStmt);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('cond_kernel', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/if\s*\(/);
    expect(kernel.source).not.toMatch(/else/);
  });
});

describe('GPUCodegen.generate — LetStmtNode', () => {
  it('emits typed variable binding', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const letVar = new VariableNode('tmp', 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], letVar);
    const letStmt = new LetStmtNode(letVar, load, store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, letStmt);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('let_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/float tmp = /);
  });

  it('uses the let variable dtype, not the kernel default dtype', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const cast = new CastNode(load, 'f32', 'i32');
    const intVar = new VariableNode('q', 'i32');
    const back = new CastNode(intVar, 'i32', 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], back);
    const letStmt = new LetStmtNode(intVar, cast, store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, letStmt);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('let_int_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/int q = /);
    expect(kernel.source).not.toMatch(/float q = /);
  });
});

describe('GPUCodegen.generate — WhileNode', () => {
  it('emits while loop', () => {
    const outBuf = buf('y', [1], 'f32');
    const condVar = new VariableNode('running', 'i32');
    const condBody = new BufferStoreNode(outBuf, [new IntImmNode(0)], new FloatImmNode(0));
    const loopBody = new BufferStoreNode(outBuf, [new IntImmNode(0)], new FloatImmNode(1));
    const whileNode = new WhileNode(condVar, condBody, loopBody);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('while_kernel', ['p0'], whileNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/while \(running\)/);
  });
});

describe('GPUCodegen.generate — BlockNode with iterVars', () => {
  it('emits const int for each bound iter var', () => {
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('vi')], new FloatImmNode(42));
    const block = new BlockNode('blk', [
      { iterVar: idx('vi'), binding: new MathOpNode('+', idx('bid'), idx('tid')) }
    ], [], [{ buffer: outBuf }], store);

    const inner = threadFor(idx('tid'), 4, 'threadIdx.x', block);
    const outer = threadFor(idx('bid'), 1, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('iter_bind', ['p0'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/const int vi = \(bid \+ tid\)/);
  });
});

describe('GPUCodegen.generate — BlockNode with initBody', () => {
  it('emits initBody before main body', () => {
    const accBuf = buf('acc', [4], 'f32');
    const inBuf = buf('inp', [4, 8], 'f32');

    const initStore = new BufferStoreNode(accBuf, [idx('i')], new FloatImmNode(0));
    const accLoad = new BufferLoadNode(accBuf, [idx('i')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const add = new MathOpNode('+', accLoad, inLoad);
    const accStore = new BufferStoreNode(accBuf, [idx('i')], add);
    const innerFor = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, accStore);

    const block = new BlockNode('reduce', [], [{ buffer: inBuf }], [{ buffer: accBuf }], innerFor, initStore);
    const outerFor = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, block);

    const bufferMap = new Map([['p0', inBuf], ['p1', accBuf]]);
    const pf = makePrimFunc('reduce_kernel', ['p0', 'p1'], outerFor, bufferMap);

    const kernel = makeCodegen().generate(pf);
    const source = kernel.source;
    const initIdx = source.indexOf('0f;');
    const addIdx = source.indexOf('acc[i]', initIdx + 1);
    expect(initIdx).toBeGreaterThan(-1);
    expect(addIdx).toBeGreaterThan(initIdx);
  });
});

describe('GPUCodegen.generate — dim clamping', () => {
  it('clamps blockDim to target max', () => {
    const target = GPUTarget({ maxBlockDimX: 512 });
    const cg = new GPUCodegen(target);

    const outBuf = buf('out', [2048], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));
    const body = threadFor(idx('tid'), 2048, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('clamp_test', ['p0'], body, bufferMap);

    const kernel = cg.generate(pf);
    expect(kernel.blockDim[0]).toBe(512);
  });

  it('clamps gridDim to target max', () => {
    const target = GPUTarget({ maxGridDimY: 100 });
    const cg = new GPUCodegen(target);

    const outBuf = buf('out', [200], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('by')], new FloatImmNode(0));
    const body = threadFor(idx('by'), 200, 'blockIdx.y', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('grid_clamp', ['p0'], body, bufferMap);

    const kernel = cg.generate(pf);
    expect(kernel.gridDim[1]).toBe(100);
  });
});

describe('GPUCodegen.generate — SeqNode', () => {
  it('visits all statements in order', () => {
    const outBuf = buf('y', [4], 'f32');
    const store1 = new BufferStoreNode(outBuf, [new IntImmNode(0)], new FloatImmNode(1));
    const store2 = new BufferStoreNode(outBuf, [new IntImmNode(1)], new FloatImmNode(2));
    const store3 = new BufferStoreNode(outBuf, [new IntImmNode(2)], new FloatImmNode(3));
    const seq = new SeqNode([store1, store2, store3]);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('seq_kernel', ['p0'], seq, bufferMap);

    const kernel = makeCodegen().generate(pf);
    const src = kernel.source;
    const idx1 = src.indexOf('1f');
    const idx2 = src.indexOf('2f');
    const idx3 = src.indexOf('3f');
    expect(idx1).toBeGreaterThan(-1);
    expect(idx2).toBeGreaterThan(idx1);
    expect(idx3).toBeGreaterThan(idx2);
  });
});

describe('GPUCodegen.generate — full elementwise GPU kernel', () => {
  it('generates a correct CUDA kernel for vector add with thread bindings', () => {
    const aBuf = buf('a', [1024], 'f32');
    const bBuf = buf('b', [1024], 'f32');
    const outBuf = buf('c', [1024], 'f32');

    const loadA = new BufferLoadNode(aBuf, [
      new MathOpNode('+', new MathOpNode('*', idx('bid'), new IntImmNode(256)), idx('tid'))
    ]);
    const loadB = new BufferLoadNode(bBuf, [
      new MathOpNode('+', new MathOpNode('*', idx('bid'), new IntImmNode(256)), idx('tid'))
    ]);
    const add = new MathOpNode('+', loadA, loadB);
    const store = new BufferStoreNode(outBuf, [
      new MathOpNode('+', new MathOpNode('*', idx('bid'), new IntImmNode(256)), idx('tid'))
    ], add);

    const block = new BlockNode('add_blk', [], [{ buffer: aBuf }, { buffer: bBuf }], [{ buffer: outBuf }], store);
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', block);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', aBuf], ['p1', bBuf], ['p2', outBuf]]);
    const pf = makePrimFunc('vec_add', ['p0', 'p1', 'p2'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);

    expect(kernel).toBeInstanceOf(GPUKernel);
    expect(kernel.name).toBe('vec_add');
    expect(kernel.blockDim).toEqual([256, 1, 1]);
    expect(kernel.gridDim).toEqual([4, 1, 1]);
    expect(kernel.params).toEqual(['a', 'b', 'c']);
    expect(kernel.source).toMatch(/__global__ void vec_add/);
    expect(kernel.source).toMatch(/const int tid = threadIdx\.x/);
    expect(kernel.source).toMatch(/const int bid = blockIdx\.x/);
    expect(kernel.source).toMatch(/c\[/);
    expect(kernel.source).toMatch(/a\[/);
    expect(kernel.source).toMatch(/b\[/);
  });
});

describe('GPUCodegen.generate — tiled matmul pattern', () => {
  it('generates correct structure for a tiled matmul with shared memory', () => {
    const aBuf = buf('A', [64, 64], 'f32');
    const bBuf = buf('B', [64, 64], 'f32');
    const cBuf = buf('C', [64, 64], 'f32');
    const smA = buf('smA', [16, 16], 'f32', 'shared');
    const smB = buf('smB', [16, 16], 'f32', 'shared');

    const initStore = new BufferStoreNode(cBuf, [idx('ty'), idx('tx')], new FloatImmNode(0));

    const loadA = new BufferLoadNode(aBuf, [idx('ty'), idx('k')]);
    const storeSmA = new BufferStoreNode(smA, [idx('ty'), idx('k')], loadA);
    const loadB = new BufferLoadNode(bBuf, [idx('k'), idx('tx')]);
    const storeSmB = new BufferStoreNode(smB, [idx('k'), idx('tx')], loadB);

    const loadSmA = new BufferLoadNode(smA, [idx('ty'), idx('kk')]);
    const loadSmB = new BufferLoadNode(smB, [idx('kk'), idx('tx')]);
    const mul = new MathOpNode('*', loadSmA, loadSmB);
    const loadC = new BufferLoadNode(cBuf, [idx('ty'), idx('tx')]);
    const accum = new MathOpNode('+', loadC, mul);
    const storeC = new BufferStoreNode(cBuf, [idx('ty'), idx('tx')], accum);

    const innerLoop = new ForNode(idx('kk'), new IntImmNode(0), new IntImmNode(16), ForKind.SERIAL, storeC);
    const tileBody = new SeqNode([storeSmA, storeSmB, innerLoop]);
    const tileLoop = new ForNode(idx('k'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, tileBody);

    const block = new BlockNode('matmul_blk', [],
      [{ buffer: aBuf }, { buffer: bBuf }],
      [{ buffer: cBuf }],
      tileLoop, initStore);

    const allocB = new AllocateNode(smB, 'shared', block);
    const allocA = new AllocateNode(smA, 'shared', allocB);

    const bindTx = threadFor(idx('tx'), 16, 'threadIdx.x', allocA);
    const bindTy = threadFor(idx('ty'), 16, 'threadIdx.y', bindTx);
    const bindBx = threadFor(idx('bx'), 4, 'blockIdx.x', bindTy);
    const bindBy = threadFor(idx('by'), 4, 'blockIdx.y', bindBx);

    const bufferMap = new Map([['p0', aBuf], ['p1', bBuf], ['p2', cBuf]]);
    const pf = makePrimFunc('matmul_tiled', ['p0', 'p1', 'p2'], bindBy, bufferMap);

    const kernel = makeCodegen().generate(pf);

    expect(kernel.blockDim).toEqual([16, 16, 1]);
    expect(kernel.gridDim).toEqual([4, 4, 1]);
    expect(kernel.sharedMemBytes).toBe(16 * 16 * 4 * 2);
    expect(kernel.source).toMatch(/__shared__ float smA\[256\]/);
    expect(kernel.source).toMatch(/__shared__ float smB\[256\]/);
    expect(kernel.source).toMatch(/for \(int kk = 0; kk < 16; kk\+\+\)/);
    expect(kernel.source).toMatch(/for \(int k = 0; k < 4; k\+\+\)/);
  });
});

describe('GPUCodegen.generate — dtype handling in generated code', () => {
  it('uses __half for f16 buffers', () => {
    const inBuf = buf('x', [4], 'f16');
    const outBuf = buf('y', [4], 'f16');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('f16_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/__half\* x/);
    expect(kernel.source).toMatch(/__half\* y/);
  });

  it('uses int for i32 buffers', () => {
    const inBuf = buf('x', [4], 'i32');
    const outBuf = buf('y', [4], 'i32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('i32_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/int\* x/);
    expect(kernel.source).toMatch(/int\* y/);
  });
});

describe('GPUCodegen.generate — dynamic shapes', () => {
  it('resolves shape params for dynamic strides', () => {
    const inBuf = buf('x', [-1, -1], 'f32');
    const outBuf = buf('y', [-1, -1], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i'), idx('j')]);
    const store = new BufferStoreNode(outBuf, [idx('i'), idx('j')], load);
    const inner = new ForNode(idx('j'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);
    const outer = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, inner);

    const nVar = new VariableNode('N', 'index');
    const mVar = new VariableNode('M', 'index');
    const shapeParamMap = new Map([
      ['x:0', nVar], ['x:1', mVar],
      ['y:0', nVar], ['y:1', mVar]
    ]);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('dynamic_kernel', ['p0', 'p1'], outer, bufferMap, [nVar, mVar], shapeParamMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/x\[i \* M \+ j\]/);
    expect(kernel.source).toMatch(/y\[i \* M \+ j\]/);
    expect(kernel.params).toContain('N');
    expect(kernel.params).toContain('M');
  });
});

describe('GPUCodegen.generate — CastNode in kernel', () => {
  it('emits C-style cast', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'i32');

    const load = new BufferLoadNode(inBuf, [idx('i')]);
    const cast = new CastNode(load, 'f32', 'i32');
    const store = new BufferStoreNode(outBuf, [idx('i')], cast);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('cast_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/\(\(int\)/);
  });
});

describe('GPUCodegen.generate — reduction with thread bindings', () => {
  it('generates reduction kernel with spatial parallelism and serial reduction', () => {
    const inBuf = buf('inp', [32, 64], 'f32');
    const outBuf = buf('out', [32], 'f32');

    const initStore = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));

    const accLoad = new BufferLoadNode(outBuf, [idx('tid')]);
    const inLoad = new BufferLoadNode(inBuf, [idx('tid'), idx('k')]);
    const add = new MathOpNode('+', accLoad, inLoad);
    const accStore = new BufferStoreNode(outBuf, [idx('tid')], add);

    const innerFor = new ForNode(idx('k'), new IntImmNode(0), new IntImmNode(64), ForKind.SERIAL, accStore);
    const block = new BlockNode('reduce', [], [{ buffer: inBuf }], [{ buffer: outBuf }], innerFor, initStore);

    const bindTid = threadFor(idx('tid'), 32, 'threadIdx.x', block);
    const bindBid = threadFor(idx('bid'), 1, 'blockIdx.x', bindTid);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('reduce_sum', ['p0', 'p1'], bindBid, bufferMap);

    const kernel = makeCodegen().generate(pf);

    expect(kernel.blockDim[0]).toBe(32);
    expect(kernel.source).toMatch(/const int tid = threadIdx\.x/);
    expect(kernel.source).toMatch(/for \(int k = 0; k < 64; k\+\+\)/);
    expect(kernel.source).not.toMatch(/for.*tid/);
    expect(kernel.source).toMatch(/out\[tid\]/);
  });
});

describe('GPUCodegen.generate — extern math calls in kernel', () => {
  it('generates correct math function calls in CUDA code', () => {
    const inBuf = buf('x', [256], 'f32');
    const outBuf = buf('y', [256], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const expCall = new CallExternNode('exp', [load], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], expCall);
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('exp_kernel', ['p0', 'p1'], body, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/expf\(x\[tid\]\)/);
  });
});

describe('GPUCodegen.generate — complex expression nesting', () => {
  it('generates correct code for fused multiply-add pattern', () => {
    const aBuf = buf('a', [256], 'f32');
    const bBuf = buf('b', [256], 'f32');
    const cBuf = buf('c', [256], 'f32');
    const outBuf = buf('out', [256], 'f32');

    const loadA = new BufferLoadNode(aBuf, [idx('i')]);
    const loadB = new BufferLoadNode(bBuf, [idx('i')]);
    const loadC = new BufferLoadNode(cBuf, [idx('i')]);
    const mul = new MathOpNode('*', loadA, loadB);
    const fma = new MathOpNode('+', mul, loadC);
    const store = new BufferStoreNode(outBuf, [idx('i')], fma);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(256), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', aBuf], ['p1', bBuf], ['p2', cBuf], ['p3', outBuf]]);
    const pf = makePrimFunc('fma_kernel', ['p0', 'p1', 'p2', 'p3'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/out\[i\] = \(\(a\[i\] \* b\[i\]\) \+ c\[i\]\)/);
  });
});

describe('GPUCodegen — state reset between generate calls', () => {
  it('does not leak state between successive calls', () => {
    const cg = makeCodegen();

    const buf1 = buf('x', [256], 'f32');
    const out1 = buf('y', [256], 'f32');
    const smem1 = buf('sm', [32], 'f32', 'shared');
    const store1 = new BufferStoreNode(out1, [idx('tid')], new BufferLoadNode(buf1, [idx('tid')]));
    const alloc1 = new AllocateNode(smem1, 'shared', store1);
    const body1 = threadFor(idx('tid'), 256, 'threadIdx.x', alloc1);
    const bufferMap1 = new Map([['p0', buf1], ['p1', out1]]);
    const pf1 = makePrimFunc('kernel1', ['p0', 'p1'], body1, bufferMap1);
    const k1 = cg.generate(pf1);

    const buf2 = buf('a', [64], 'f32');
    const out2 = buf('b', [64], 'f32');
    const store2 = new BufferStoreNode(out2, [idx('t')], new BufferLoadNode(buf2, [idx('t')]));
    const body2 = threadFor(idx('t'), 64, 'threadIdx.x', store2);
    const bufferMap2 = new Map([['p0', buf2], ['p1', out2]]);
    const pf2 = makePrimFunc('kernel2', ['p0', 'p1'], body2, bufferMap2);
    const k2 = cg.generate(pf2);

    expect(k1.blockDim[0]).toBe(256);
    expect(k2.blockDim[0]).toBe(64);
    expect(k1.sharedMemBytes).toBe(32 * 4);
    expect(k2.sharedMemBytes).toBe(0);
    expect(k1.source).toMatch(/__shared__/);
    expect(k2.source).not.toMatch(/__shared__/);
  });
});
