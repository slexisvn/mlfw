import { describe, it, expect } from 'vitest';
import { WebGPUCodegen, WebGPUKernel } from '../../../src/backend/webgpu/codegen.js';
import { WebGPUTarget } from '../../../src/backend/target.js';
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
  return new WebGPUCodegen(WebGPUTarget(overrides));
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

describe('WebGPUCodegen._exprToWGSL', () => {
  function exprToWGSL(node) {
    const cg = makeCodegen();
    cg._defaultDtype = 'f32';
    return cg._exprToWGSL(node);
  }

  it('renders IntImmNode', () => {
    expect(exprToWGSL(new IntImmNode(42))).toBe('42');
    expect(exprToWGSL(new IntImmNode(0))).toBe('0');
    expect(exprToWGSL(new IntImmNode(-7))).toBe('-7');
  });

  it('renders FloatImmNode without suffix', () => {
    expect(exprToWGSL(new FloatImmNode(3.14))).toBe('3.14');
  });

  it('renders integer float with decimal point', () => {
    expect(exprToWGSL(new FloatImmNode(1))).toBe('1.0');
    expect(exprToWGSL(new FloatImmNode(0))).toBe('0.0');
  });

  it('renders Infinity as WGSL hex float', () => {
    expect(exprToWGSL(new FloatImmNode(Infinity))).toMatch(/f32\(/);
    expect(exprToWGSL(new FloatImmNode(-Infinity))).toMatch(/f32\(/);
  });

  it('renders VariableNode', () => {
    expect(exprToWGSL(new VariableNode('tid', 'index'))).toBe('tid');
  });

  it('renders binary MathOpNode', () => {
    const add = new MathOpNode('+', new VariableNode('a', 'f32'), new VariableNode('b', 'f32'));
    expect(exprToWGSL(add)).toBe('(a + b)');

    const mul = new MathOpNode('*', new VariableNode('i', 'index'), new IntImmNode(8));
    expect(exprToWGSL(mul)).toBe('(i * 8)');
  });

  it('renders unary MathOpNode (neg)', () => {
    const neg = new MathOpNode('-', new VariableNode('x', 'f32'));
    expect(exprToWGSL(neg)).toBe('(-x)');
  });

  it('renders integer division', () => {
    const div = new MathOpNode('//', new VariableNode('x', 'index'), new IntImmNode(4));
    expect(exprToWGSL(div)).toBe('(x / 4)');
  });

  it('renders modulo', () => {
    const mod = new MathOpNode('%', new VariableNode('x', 'index'), new IntImmNode(4));
    expect(exprToWGSL(mod)).toBe('(x % 4)');
  });

  it('renders nested expressions', () => {
    const inner = new MathOpNode('+', new VariableNode('i', 'index'), new IntImmNode(1));
    const outer = new MathOpNode('*', inner, new IntImmNode(16));
    expect(exprToWGSL(outer)).toBe('((i + 1) * 16)');
  });

  it('renders CompareNode', () => {
    const ge = new CompareNode('ge', new VariableNode('x', 'index'), new IntImmNode(0));
    expect(exprToWGSL(ge)).toBe('(x >= 0)');

    const eq = new CompareNode('eq', new VariableNode('a', 'index'), new VariableNode('b', 'index'));
    expect(exprToWGSL(eq)).toBe('(a == b)');

    const ne = new CompareNode('ne', new VariableNode('a', 'index'), new IntImmNode(0));
    expect(exprToWGSL(ne)).toBe('(a != 0)');
  });

  it('renders IfThenElseNode as select()', () => {
    const cond = new CompareNode('gt', new VariableNode('x', 'f32'), new FloatImmNode(0));
    const node = new IfThenElseNode(cond, new FloatImmNode(1), new FloatImmNode(0));
    const result = exprToWGSL(node);
    expect(result).toMatch(/select\(/);
    expect(result).toBe('select(0.0, 1.0, (x > 0.0))');
  });

  it('renders CastNode as type constructor', () => {
    const node = new CastNode(new FloatImmNode(3.7), 'f32', 'i32');
    expect(exprToWGSL(node)).toBe('i32(3.7)');
  });

  it('renders CastNode to f32', () => {
    const node = new CastNode(new VariableNode('i', 'index'), 'i32', 'f32');
    expect(exprToWGSL(node)).toBe('f32(i)');
  });

  it('renders BufferLoadNode with flat index', () => {
    const b = buf('data', [4], 'f32');
    const node = new BufferLoadNode(b, [new VariableNode('i', 'index')]);
    expect(exprToWGSL(node)).toBe('data[i]');
  });

  it('renders BufferLoadNode with multi-dim index', () => {
    const b = buf('mat', [3, 4], 'f32');
    const node = new BufferLoadNode(b, [idx('i'), idx('j')]);
    expect(exprToWGSL(node)).toBe('mat[i * 4 + j]');
  });

  it('returns 0 for null input', () => {
    expect(exprToWGSL(null)).toBe('0');
  });

  it('returns 0 for unknown node types', () => {
    expect(exprToWGSL({ type: 'UnknownNode' })).toBe('0');
  });
});

describe('WebGPUCodegen._emitExternCall', () => {
  function externCall(name, args, dtype = 'f32') {
    const cg = makeCodegen();
    cg._defaultDtype = dtype;
    return cg._emitExternCall(new CallExternNode(name, args, dtype));
  }

  it('maps math functions without suffix', () => {
    const x = new VariableNode('x', 'f32');
    expect(externCall('exp', [x])).toBe('exp(x)');
    expect(externCall('log', [x])).toBe('log(x)');
    expect(externCall('sqrt', [x])).toBe('sqrt(x)');
    expect(externCall('tanh', [x])).toBe('tanh(x)');
    expect(externCall('sin', [x])).toBe('sin(x)');
    expect(externCall('cos', [x])).toBe('cos(x)');
    expect(externCall('abs', [x])).toBe('abs(x)');
    expect(externCall('ceil', [x])).toBe('ceil(x)');
    expect(externCall('floor', [x])).toBe('floor(x)');
  });

  it('maps rsqrt to inverseSqrt', () => {
    const x = new VariableNode('x', 'f32');
    expect(externCall('rsqrt', [x])).toBe('inverseSqrt(x)');
  });

  it('maps sign to sign', () => {
    const x = new VariableNode('x', 'f32');
    expect(externCall('sign', [x])).toBe('sign(x)');
  });

  it('handles two-argument functions', () => {
    const a = new VariableNode('a', 'f32');
    const b = new VariableNode('b', 'f32');
    expect(externCall('max', [a, b])).toBe('max(a, b)');
    expect(externCall('min', [a, b])).toBe('min(a, b)');
    expect(externCall('pow', [a, b])).toBe('pow(a, b)');
  });

  it('maps fmod to inline modulo', () => {
    const a = new VariableNode('a', 'f32');
    const b = new VariableNode('b', 'f32');
    expect(externCall('fmod', [a, b])).toBe('(a % b)');
  });
});

describe('WebGPUCodegen._flatIndex', () => {
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

  it('handles stride-1 dimension without multiplication', () => {
    const b = buf('v', [4, 1]);
    const result = flatIndex(b, [idx('i'), idx('j')]);
    expect(result).toBe('i + j');
  });
});

describe('WebGPUCodegen._scanBindings', () => {
  it('extracts threadIdx.x binding and sets workgroupSize', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [256]), [idx('tid')], new FloatImmNode(0));
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    cg._scanBindings(body);
    expect(cg._threadBindings.has('threadIdx.x')).toBe(true);
    expect(cg._threadBindings.get('threadIdx.x')[0].varName).toBe('tid');
    expect(cg._workgroupSize[0]).toBe(256);
  });

  it('extracts blockIdx.x binding and sets dispatchSize', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [1024]), [idx('bid')], new FloatImmNode(0));
    const body = threadFor(idx('bid'), 64, 'blockIdx.x', store);
    cg._scanBindings(body);
    expect(cg._threadBindings.has('blockIdx.x')).toBe(true);
    expect(cg._dispatchSize[0]).toBe(64);
  });

  it('extracts threadIdx.y and sets workgroupSize[1]', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [16]), [idx('ty')], new FloatImmNode(0));
    const body = threadFor(idx('ty'), 16, 'threadIdx.y', store);
    cg._scanBindings(body);
    expect(cg._workgroupSize[1]).toBe(16);
  });

  it('finds nested thread bindings', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [1024]), [idx('tid')], new FloatImmNode(0));
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);
    cg._scanBindings(outer);
    expect(cg._workgroupSize[0]).toBe(256);
    expect(cg._dispatchSize[0]).toBe(4);
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

  it('handles dynamic extent', () => {
    const cg = makeCodegen();
    const store = new BufferStoreNode(buf('out', [256]), [idx('tid')], new FloatImmNode(0));
    const dynExtent = new VariableNode('N', 'index');
    const body = new ForNode(idx('tid'), new IntImmNode(0), dynExtent, ForKind.THREAD_BINDING, store, 'threadIdx.x');
    cg._scanBindings(body);
    expect(cg._threadBindings.get('threadIdx.x')[0].isDynamic).toBe(true);
    expect(cg._workgroupSize[0]).toBe(1);
  });
});

describe('WebGPUCodegen._applyBindingDim', () => {
  it('maps threadIdx.x to workgroupSize[0]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.x', 128);
    expect(cg._workgroupSize).toEqual([128, 1, 1]);
  });

  it('maps threadIdx.y to workgroupSize[1]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.y', 16);
    expect(cg._workgroupSize).toEqual([1, 16, 1]);
  });

  it('maps threadIdx.z to workgroupSize[2]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.z', 4);
    expect(cg._workgroupSize).toEqual([1, 1, 4]);
  });

  it('maps blockIdx.x to dispatchSize[0]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('blockIdx.x', 32);
    expect(cg._dispatchSize).toEqual([32, 1, 1]);
  });

  it('maps blockIdx.y to dispatchSize[1]', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('blockIdx.y', 64);
    expect(cg._dispatchSize).toEqual([1, 64, 1]);
  });

  it('ignores tags without dot separator', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('noprefix', 100);
    expect(cg._workgroupSize).toEqual([1, 1, 1]);
    expect(cg._dispatchSize).toEqual([1, 1, 1]);
  });

  it('ignores invalid axis character', () => {
    const cg = makeCodegen();
    cg._applyBindingDim('threadIdx.w', 100);
    expect(cg._workgroupSize).toEqual([1, 1, 1]);
  });
});

describe('WebGPUCodegen.generate — buffer bindings', () => {
  it('emits @group(0) @binding(N) var<storage> for buffers', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/@group\(0\) @binding\(0\) var<storage, read> x: array<f32>/);
    expect(kernel.source).toMatch(/@group\(0\) @binding\(1\) var<storage, read_write> y: array<f32>/);
    expect(kernel.name).toBe('copy_kernel');
    expect(kernel.params).toEqual(['x', 'y']);
  });

  it('classifies read-only vs read-write buffers', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('copy_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.bindings[0].mode).toBe('read');
    expect(kernel.bindings[1].mode).toBe('read_write');
  });

  it('returns WebGPUKernel instance', () => {
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('fill_kernel', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel).toBeInstanceOf(WebGPUKernel);
  });
});

describe('WebGPUCodegen.generate — shape params', () => {
  it('emits ShapeParams struct and uniform binding', () => {
    const inBuf = buf('x', [4], 'f32');
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const nVar = new VariableNode('N', 'index');
    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('dynamic_kernel', ['p0', 'p1'], forNode, bufferMap, [nVar]);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/struct ShapeParams/);
    expect(kernel.source).toMatch(/N: u32/);
    expect(kernel.source).toMatch(/@group\(0\) @binding\(2\) var<uniform> _shapes: ShapeParams/);
    expect(kernel.params).toContain('N');
  });

  it('resolves shape param variables to _shapes.name', () => {
    const cg = makeCodegen();
    const nVar = new VariableNode('N', 'index');
    cg._primFunc = { shapeParams: [nVar] };
    expect(cg._resolveVariable('N')).toBe('i32(_shapes.N)');
  });

  it('does not wrap non-shape-param variables', () => {
    const cg = makeCodegen();
    cg._primFunc = { shapeParams: [] };
    expect(cg._resolveVariable('tid')).toBe('tid');
  });
});

describe('WebGPUCodegen.generate — thread bindings', () => {
  it('emits let assignments with WGSL builtins', () => {
    const inBuf = buf('x', [1024], 'f32');
    const outBuf = buf('y', [1024], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const store = new BufferStoreNode(outBuf, [idx('tid')], load);
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('thread_test', ['p0', 'p1'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/let tid: i32 = i32\(_lid\.x\)/);
    expect(kernel.source).toMatch(/let bid: i32 = i32\(_wid\.x\)/);
    expect(kernel.workgroupSize).toEqual([256, 1, 1]);
    expect(kernel.dispatchSize).toEqual([4, 1, 1]);
  });

  it('sets 2D workgroup/dispatch dimensions', () => {
    const outBuf = buf('out', [16, 16], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('ty'), idx('tx')], new FloatImmNode(0));
    const bindTx = threadFor(idx('tx'), 16, 'threadIdx.x', store);
    const bindTy = threadFor(idx('ty'), 16, 'threadIdx.y', bindTx);
    const bindBx = threadFor(idx('bx'), 4, 'blockIdx.x', bindTy);
    const bindBy = threadFor(idx('by'), 4, 'blockIdx.y', bindBx);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('grid2d', ['p0'], bindBy, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.workgroupSize).toEqual([16, 16, 1]);
    expect(kernel.dispatchSize).toEqual([4, 4, 1]);
  });

  it('does not emit for loop for thread binding', () => {
    const outBuf = buf('y', [256], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('no_loop', ['p0'], inner, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).not.toMatch(/for \(var tid/);
    expect(kernel.source).toMatch(/let tid/);
  });
});

describe('WebGPUCodegen.generate — kernel structure', () => {
  it('emits @compute @workgroup_size annotation', () => {
    const outBuf = buf('y', [256], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('wg_test', ['p0'], body, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/@compute @workgroup_size\(256, 1, 1\)/);
    expect(kernel.source).toMatch(/fn wg_test\(/);
  });

  it('includes @builtin parameters in fn signature', () => {
    const outBuf = buf('y', [1024], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('builtin_test', ['p0'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/@builtin\(local_invocation_id\) _lid: vec3u/);
    expect(kernel.source).toMatch(/@builtin\(workgroup_id\) _wid: vec3u/);
  });

  it('does not contain CUDA-isms', () => {
    const inBuf = buf('x', [256], 'f32');
    const outBuf = buf('y', [256], 'f32');
    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const store = new BufferStoreNode(outBuf, [idx('tid')], load);
    const body = threadFor(idx('tid'), 256, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('no_cuda', ['p0', 'p1'], body, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).not.toMatch(/__global__/);
    expect(kernel.source).not.toMatch(/__shared__/);
    expect(kernel.source).not.toMatch(/float\*/);
    expect(kernel.source).not.toMatch(/int\*/);
    expect(kernel.source).not.toMatch(/#pragma/);
    expect(kernel.source).not.toMatch(/threadIdx\./);
    expect(kernel.source).not.toMatch(/blockIdx\./);
  });
});

describe('WebGPUCodegen.generate — for loops', () => {
  it('emits WGSL-style for loop', () => {
    const outBuf = buf('y', [8], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('loop_test', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/for \(var i: i32 = 0; i < 8; i = i \+ 1\)/);
  });

  it('emits unrolled loop same as serial (no pragma)', () => {
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.UNROLLED, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('unroll_test', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/for \(var i: i32 = 0; i < 4; i = i \+ 1\)/);
    expect(kernel.source).not.toMatch(/#pragma/);
  });
});

describe('WebGPUCodegen.generate — if/else', () => {
  it('emits if/else blocks', () => {
    const outBuf = buf('y', [4], 'f32');
    const cond = new CompareNode('lt', idx('i'), new IntImmNode(2));
    const thenStore = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(1));
    const elseStore = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const ifNode = new IfThenElseNode(cond, thenStore, elseStore);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, ifNode);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('if_test', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/if \(/);
    expect(kernel.source).toMatch(/\} else \{/);
  });
});

describe('WebGPUCodegen.generate — shared memory', () => {
  it('emits var<workgroup> for shared buffers', () => {
    const sharedBuf = buf('smem', [128], 'f32', 'shared');
    const outBuf = buf('out', [128], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(sharedBuf, [idx('i')]));
    const alloc = new AllocateNode(sharedBuf, 'shared', store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(128), ForKind.SERIAL, alloc);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('shared_test', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/var<workgroup> smem: array<f32, 128>/);
    expect(kernel.source).not.toMatch(/__shared__/);
  });
});

describe('WebGPUCodegen.generate — local arrays', () => {
  it('emits var for local allocations', () => {
    const localBuf = buf('tmp', [8], 'f32', 'local');
    const outBuf = buf('out', [8], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(localBuf, [idx('i')]));
    const alloc = new AllocateNode(localBuf, 'local', store);
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, alloc);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('local_test', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/var tmp: array<f32, 8>/);
  });
});

describe('WebGPUCodegen.generate — let statements', () => {
  it('emits var with type for let bindings', () => {
    const outBuf = buf('y', [4], 'f32');
    const value = new MathOpNode('*', new FloatImmNode(2), new FloatImmNode(3));
    const store = new BufferStoreNode(outBuf, [new IntImmNode(0)], new VariableNode('temp', 'f32'));
    const letNode = new LetStmtNode(new VariableNode('temp', 'f32'), value, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('let_test', ['p0'], letNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/var temp: f32 = /);
  });
});

describe('WebGPUCodegen.generate — while loop', () => {
  it('emits while loop', () => {
    const outBuf = buf('y', [4], 'f32');
    const condVar = new VariableNode('running', 'bool');
    const store = new BufferStoreNode(outBuf, [new IntImmNode(0)], new FloatImmNode(0));
    const condBody = new BufferStoreNode(outBuf, [new IntImmNode(0)], new FloatImmNode(1));
    const whileNode = new WhileNode(condVar, condBody, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('while_test', ['p0'], whileNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/while \(running\)/);
  });
});

describe('WebGPUCodegen.generate — workgroup size clamping', () => {
  it('clamps workgroup size to target limits', () => {
    const outBuf = buf('y', [512], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('tid')], new FloatImmNode(0));
    const body = threadFor(idx('tid'), 512, 'threadIdx.x', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('clamp_test', ['p0'], body, bufferMap);

    const kernel = new WebGPUCodegen(WebGPUTarget({ maxBlockDimX: 256 })).generate(pf);
    expect(kernel.workgroupSize[0]).toBe(256);
  });

  it('clamps dispatch size to target limits', () => {
    const outBuf = buf('y', [1024], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('bid')], new FloatImmNode(0));
    const body = threadFor(idx('bid'), 100000, 'blockIdx.x', store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('dispatch_clamp', ['p0'], body, bufferMap);

    const kernel = new WebGPUCodegen(WebGPUTarget({ maxGridDimX: 65535 })).generate(pf);
    expect(kernel.dispatchSize[0]).toBe(65535);
  });
});

describe('WebGPUCodegen.generate — f16 support', () => {
  it('emits enable f16 directive when f16 buffers present', () => {
    const inBuf = buf('x', [4], 'f16');
    const outBuf = buf('y', [4], 'f16');
    const store = new BufferStoreNode(outBuf, [idx('i')], new BufferLoadNode(inBuf, [idx('i')]));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('f16_kernel', ['p0', 'p1'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/enable f16;/);
    expect(kernel.source).toMatch(/array<f16>/);
  });

  it('does not emit enable f16 for f32-only kernels', () => {
    const outBuf = buf('y', [4], 'f32');
    const store = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const forNode = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);

    const bufferMap = new Map([['p0', outBuf]]);
    const pf = makePrimFunc('f32_kernel', ['p0'], forNode, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).not.toMatch(/enable f16/);
  });
});

describe('WebGPUCodegen.generate — full kernel', () => {
  it('generates valid elementwise copy kernel', () => {
    const inBuf = buf('x', [1024], 'f32');
    const outBuf = buf('y', [1024], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const store = new BufferStoreNode(outBuf, [idx('tid')], load);
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('elementwise_copy', ['p0', 'p1'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);

    expect(kernel.source).toMatch(/@group\(0\) @binding\(0\)/);
    expect(kernel.source).toMatch(/@group\(0\) @binding\(1\)/);
    expect(kernel.source).toMatch(/@compute @workgroup_size\(256, 1, 1\)/);
    expect(kernel.source).toMatch(/fn elementwise_copy/);
    expect(kernel.source).toMatch(/y\[tid\] = x\[tid\]/);

    expect(kernel.workgroupSize).toEqual([256, 1, 1]);
    expect(kernel.dispatchSize).toEqual([4, 1, 1]);
    expect(kernel.bindings).toHaveLength(2);
  });

  it('generates kernel with reduction loop', () => {
    const inBuf = buf('x', [4, 8], 'f32');
    const outBuf = buf('y', [4], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('i'), idx('k')]);
    const accum = new MathOpNode('+', new VariableNode('acc', 'f32'), load);
    const store = new BufferStoreNode(outBuf, [idx('i')], new VariableNode('acc', 'f32'));
    const innerBody = new BufferStoreNode(outBuf, [idx('i')], accum);
    const inner = new ForNode(idx('k'), new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, innerBody);

    const initStore = new BufferStoreNode(outBuf, [idx('i')], new FloatImmNode(0));
    const seq = new SeqNode([initStore, inner, store]);
    const outer = new ForNode(idx('i'), new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, seq);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('reduce_sum', ['p0', 'p1'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    expect(kernel.source).toMatch(/for \(var i: i32 = 0/);
    expect(kernel.source).toMatch(/for \(var k: i32 = 0/);
  });

  it('generates balanced braces', () => {
    const inBuf = buf('x', [256], 'f32');
    const outBuf = buf('y', [256], 'f32');

    const load = new BufferLoadNode(inBuf, [idx('tid')]);
    const store = new BufferStoreNode(outBuf, [idx('tid')], load);
    const inner = threadFor(idx('tid'), 256, 'threadIdx.x', store);
    const outer = threadFor(idx('bid'), 4, 'blockIdx.x', inner);

    const bufferMap = new Map([['p0', inBuf], ['p1', outBuf]]);
    const pf = makePrimFunc('balanced', ['p0', 'p1'], outer, bufferMap);

    const kernel = makeCodegen().generate(pf);
    const opens = (kernel.source.match(/\{/g) || []).length;
    const closes = (kernel.source.match(/\}/g) || []).length;
    expect(opens).toBe(closes);
  });
});
