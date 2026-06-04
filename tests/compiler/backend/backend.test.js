import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';
import { Buffer } from '../../../src/compiler/ir/tensor/buffer.js';
import {
  PrimFunc, ForNode, BlockNode, SeqNode, BufferStoreNode, BufferLoadNode,
  VariableNode, IntImmNode, FloatImmNode, MathOpNode, BlockRealizeNode,
  ForKind, AllocateNode, CallExternNode
} from '../../../src/compiler/ir/tensor/nodes.js';
import { CPUCodegen } from '../../../src/compiler/backend/cpu/codegen.js';
import { GPUCodegen } from '../../../src/compiler/backend/gpu/codegen.js';
import { BackendPipeline } from '../../../src/compiler/backend/pipeline.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { LibrarySelector, LibraryCall, createCPULibrarySelector } from '../../../src/compiler/backend/library_selector.js';
import { RuntimeModule, RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { Schedule, resetVarCounter } from '../../../src/compiler/schedule/schedule.js';

const f32 = ScalarType.F32;
const f32_4x4 = new TensorType([4, 4], f32);

function makeSimpleFunc() {
  const A = new Buffer('A', [8], 'f32', 'global');
  const B = new Buffer('B', [8], 'f32', 'global');
  const C = new Buffer('C', [8], 'f32', 'global');
  const i = new VariableNode('i', 'int32');
  const vi = new VariableNode('vi', 'int32');
  const bind = new BlockRealizeNode(vi, i);
  const load_a = new BufferLoadNode(A, [vi]);
  const load_b = new BufferLoadNode(B, [vi]);
  const expr = new MathOpNode('+', load_a, load_b);
  const store = new BufferStoreNode(C, [vi], expr);
  const block = new BlockNode('add_block', [bind], [{buffer: A}, {buffer: B}], [{buffer: C}], store);
  const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, block);
  const pA = new VariableNode('pA', 'int32');
  const pB = new VariableNode('pB', 'int32');
  const pC = new VariableNode('pC', 'int32');
  return new PrimFunc('vec_add', [pA, pB, pC], loop, new Map([[pA, A], [pB, B], [pC, C]]));
}

function makeGPUFunc() {
  const A = new Buffer('A', [256], 'f32', 'global');
  const C = new Buffer('C', [256], 'f32', 'global');
  const bx = new VariableNode('bx', 'int32');
  const tx = new VariableNode('tx', 'int32');
  const vi = new VariableNode('vi', 'int32');
  const bind = new BlockRealizeNode(vi, new MathOpNode('+', new MathOpNode('*', bx, new IntImmNode(32)), tx));
  const load = new BufferLoadNode(A, [vi]);
  const call = new CallExternNode('exp', [load], 'f32');
  const store = new BufferStoreNode(C, [vi], call);
  const block = new BlockNode('exp_block', [bind], [{buffer: A}], [{buffer: C}], store);
  const txLoop = new ForNode(tx, new IntImmNode(0), new IntImmNode(32), ForKind.THREAD_BINDING, block, 'threadIdx.x');
  const bxLoop = new ForNode(bx, new IntImmNode(0), new IntImmNode(8), ForKind.THREAD_BINDING, txLoop, 'blockIdx.x');
  const pA = new VariableNode('pA', 'int32');
  const pC = new VariableNode('pC', 'int32');
  return new PrimFunc('exp_kernel', [pA, pC], bxLoop, new Map([[pA, A], [pC, C]]));
}

describe('CPUCodegen', () => {
  it('generates JS for simple vector add', () => {
    const func = makeSimpleFunc();
    const codegen = new CPUCodegen(CPUTarget());
    const source = codegen.generate(func);
    assert.ok(source.includes('function vec_add(A, B, C)'));
    assert.ok(source.includes('for (let i = 0; i < 8; i++)'));
    assert.ok(source.includes('C['));
    assert.ok(source.includes('A['));
    assert.ok(source.includes('+'));
  });

  it('generates valid executable JS', () => {
    const func = makeSimpleFunc();
    const source = new CPUCodegen(CPUTarget()).generate(func);
    const fn = new Function('return ' + source)();
    const A = new Float32Array([1,2,3,4,5,6,7,8]);
    const B = new Float32Array([10,20,30,40,50,60,70,80]);
    const C = new Float32Array(8);
    fn(A, B, C);
    assert.deepEqual([...C], [11,22,33,44,55,66,77,88]);
  });

  it('handles unrolled loops', () => {
    const A = new Buffer('A', [4], 'f32', 'global');
    const i = new VariableNode('i', 'int32');
    const store = new BufferStoreNode(A, [i], new FloatImmNode(0));
    const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(4), ForKind.UNROLLED, store);
    const p = new VariableNode('p', 'int32');
    const func = new PrimFunc('unrolled', [p], loop, new Map([[p, A]]));
    const source = new CPUCodegen(CPUTarget()).generate(func);
    assert.ok(source.includes('const i = 0'));
    assert.ok(source.includes('const i = 3'));
  });

  it('handles AllocateNode', () => {
    const A = new Buffer('A', [8], 'f32', 'global');
    const tmp = new Buffer('tmp', [8], 'f32', 'local');
    const i = new VariableNode('i', 'int32');
    const store = new BufferStoreNode(tmp, [i], new BufferLoadNode(A, [i]));
    const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(8), ForKind.SERIAL, store);
    const alloc = new AllocateNode(tmp, 'local', loop);
    const p = new VariableNode('p', 'int32');
    const func = new PrimFunc('with_alloc', [p], alloc, new Map([[p, A]]));
    const source = new CPUCodegen(CPUTarget()).generate(func);
    assert.ok(source.includes('const tmp = new Float32Array(8)'));
  });

  it('emits Math.* for extern calls', () => {
    const A = new Buffer('A', [4], 'f32', 'global');
    const i = new VariableNode('i', 'int32');
    const load = new BufferLoadNode(A, [i]);
    const call = new CallExternNode('exp', [load], 'f32');
    const store = new BufferStoreNode(A, [i], call);
    const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(4), ForKind.SERIAL, store);
    const p = new VariableNode('p', 'int32');
    const func = new PrimFunc('test', [p], loop, new Map([[p, A]]));
    const source = new CPUCodegen(CPUTarget()).generate(func);
    assert.ok(source.includes('Math.exp('));
  });

  it('generates from lowered graph IR', () => {
    const func = buildFunction('test', [f32_4x4, f32_4x4], [f32_4x4], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    const primFunc = lowerGraphToPrimFunc(func);
    const source = new CPUCodegen(CPUTarget()).generate(primFunc);
    assert.ok(source.includes('function test('));
    assert.ok(source.includes('for ('));
    assert.ok(source.includes('+'));
  });
});

describe('GPUCodegen', () => {
  it('generates CUDA-like kernel', () => {
    const func = makeGPUFunc();
    const kernel = new GPUCodegen(GPUTarget()).generate(func);
    assert.ok(kernel.source.includes('__global__ void exp_kernel'));
    assert.ok(kernel.source.includes('threadIdx.x'));
    assert.ok(kernel.source.includes('blockIdx.x'));
    assert.ok(kernel.source.includes('expf('));
  });

  it('extracts correct blockDim and gridDim', () => {
    const func = makeGPUFunc();
    const kernel = new GPUCodegen(GPUTarget()).generate(func);
    assert.deepEqual(kernel.blockDim, [32, 1, 1]);
    assert.deepEqual(kernel.gridDim, [8, 1, 1]);
  });

  it('skips thread-bound loops in body emission', () => {
    const func = makeGPUFunc();
    const kernel = new GPUCodegen(GPUTarget()).generate(func);
    assert.ok(!kernel.source.includes('for (int bx'));
    assert.ok(!kernel.source.includes('for (int tx'));
  });

  it('handles shared memory declarations', () => {
    const A = new Buffer('A', [256], 'f32', 'global');
    const smem = new Buffer('smem', [32], 'f32', 'shared');
    const i = new VariableNode('i', 'int32');
    const store = new BufferStoreNode(smem, [i], new BufferLoadNode(A, [i]));
    const loop = new ForNode(i, new IntImmNode(0), new IntImmNode(32), ForKind.SERIAL, store);
    const alloc = new AllocateNode(smem, 'shared', loop);
    const tx = new VariableNode('tx', 'int32');
    const txLoop = new ForNode(tx, new IntImmNode(0), new IntImmNode(32), ForKind.THREAD_BINDING, alloc, 'threadIdx.x');
    const p = new VariableNode('p', 'int32');
    const func = new PrimFunc('smem_kernel', [p], txLoop, new Map([[p, A]]));
    const kernel = new GPUCodegen(GPUTarget()).generate(func);
    assert.ok(kernel.source.includes('__shared__ float smem[32]'));
    assert.equal(kernel.sharedMemBytes, 128);
  });
});

describe('BackendPipeline', () => {
  it('compiles CPU target to JS', () => {
    const func = makeSimpleFunc();
    const pipeline = new BackendPipeline(CPUTarget());
    const result = pipeline.compile(func);
    assert.equal(result.metadata.kind, 'js');
    assert.ok(result.source.includes('function vec_add'));
  });

  it('compiles GPU target to CUDA', () => {
    const func = makeGPUFunc();
    const pipeline = new BackendPipeline(GPUTarget());
    const result = pipeline.compile(func);
    assert.equal(result.metadata.kind, 'cuda');
    assert.deepEqual(result.metadata.blockDim, [32, 1, 1]);
  });

  it('compiles multiple functions', () => {
    const funcs = [makeSimpleFunc(), makeSimpleFunc()];
    funcs[1].name = 'vec_add_2';
    const pipeline = new BackendPipeline(CPUTarget());
    const results = pipeline.compileAll(funcs);
    assert.equal(results.length, 2);
  });
});

describe('LibrarySelector', () => {
  it('selects library call for matching op', () => {
    const selector = createCPULibrarySelector(CPUTarget());
    const call = selector.select('dot', [64, 64], 'f32');
    assert.ok(call);
    assert.equal(call.libraryName, 'blas');
    assert.equal(call.functionName, 'sgemm');
  });

  it('rejects small tensors below minElements', () => {
    const selector = createCPULibrarySelector(CPUTarget());
    const call = selector.select('dot', [2, 2], 'f32');
    assert.equal(call, null);
  });

  it('rejects unknown ops', () => {
    const selector = createCPULibrarySelector(CPUTarget());
    assert.equal(selector.select('unknown_op', [64], 'f32'), null);
  });

  it('rejects incompatible dtype', () => {
    const selector = createCPULibrarySelector(CPUTarget());
    const call = selector.select('dot', [64, 64], 'i32');
    assert.equal(call, null);
  });
});

describe('RuntimeModule', () => {
  it('registers and executes CPU kernel', () => {
    const func = makeSimpleFunc();
    const pipeline = new BackendPipeline(CPUTarget());
    const compiled = pipeline.compile(func);
    const mod = new RuntimeModule('test_mod');
    mod.addCompiledKernel(compiled);
    assert.ok(mod.listKernels().includes('vec_add'));

    const A = RuntimeTensor.fromArray([1,2,3,4,5,6,7,8], [8]);
    const B = RuntimeTensor.fromArray([10,20,30,40,50,60,70,80], [8]);
    const C = RuntimeTensor.zeros([8]);
    mod.run('vec_add', A, B, C);
    assert.deepEqual([...C.data], [11,22,33,44,55,66,77,88]);
  });

  it('serializes module', () => {
    const func = makeSimpleFunc();
    const compiled = new BackendPipeline(CPUTarget()).compile(func);
    const mod = new RuntimeModule('test');
    mod.addCompiledKernel(compiled);
    const serialized = mod.serialize();
    assert.equal(serialized.name, 'test');
    assert.equal(serialized.kernels.length, 1);
    assert.equal(serialized.kernels[0].name, 'vec_add');
  });

  it('throws on missing kernel', () => {
    const mod = new RuntimeModule('test');
    assert.throws(() => mod.run('nonexistent'), /not found/);
  });
});

describe('RuntimeTensor', () => {
  it('creates zeros', () => {
    const t = RuntimeTensor.zeros([2, 3]);
    assert.equal(t.numel, 6);
    assert.equal(t.rank, 2);
    assert.deepEqual(t.shape, [2, 3]);
    assert.deepEqual(t.strides, [3, 1]);
    assert.equal(t.get([0, 0]), 0);
  });

  it('get and set', () => {
    const t = RuntimeTensor.zeros([3, 4]);
    t.set([1, 2], 42);
    assert.equal(t.get([1, 2]), 42);
    assert.equal(t.get([0, 0]), 0);
  });

  it('fromArray', () => {
    const t = RuntimeTensor.fromArray([1, 2, 3, 4], [2, 2]);
    assert.equal(t.get([0, 0]), 1);
    assert.equal(t.get([0, 1]), 2);
    assert.equal(t.get([1, 0]), 3);
    assert.equal(t.get([1, 1]), 4);
  });
});

describe('End-to-end: Graph IR -> TensorIR -> Codegen -> Execute', () => {
  it('elementwise add', () => {
    const func = buildFunction('e2e_add', [new TensorType([4], f32), new TensorType([4], f32)], [new TensorType([4], f32)], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    const primFunc = lowerGraphToPrimFunc(func);
    const compiled = new BackendPipeline(CPUTarget()).compile(primFunc);
    const mod = new RuntimeModule('e2e');
    mod.addCompiledKernel(compiled);
    const A = RuntimeTensor.fromArray([1, 2, 3, 4], [4]);
    const B = RuntimeTensor.fromArray([10, 20, 30, 40], [4]);
    const C = RuntimeTensor.zeros([4]);
    mod.run('e2e_add', A, B, C);
    assert.deepEqual([...C.data], [11, 22, 33, 44]);
  });

  it('elementwise mul + exp chain', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('e2e_chain', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      const m = b.mul(x, y);
      b.returnOp([b.exp(m.getResult(0)).getResult(0)]);
    });
    const primFunc = lowerGraphToPrimFunc(func);
    const compiled = new BackendPipeline(CPUTarget()).compile(primFunc);
    const mod = new RuntimeModule('e2e');
    mod.addCompiledKernel(compiled);
    const A = RuntimeTensor.fromArray([0, 0, 0, 0], [4]);
    const B = RuntimeTensor.fromArray([1, 2, 3, 4], [4]);
    const C = RuntimeTensor.zeros([4]);
    mod.run('e2e_chain', A, B, C);
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(C.data[i] - Math.exp(0)) < 1e-6);
    }
  });

  it('matmul 2x3 @ 3x2', () => {
    const func = buildFunction('e2e_dot', [new TensorType([2, 3], f32), new TensorType([3, 2], f32)], [new TensorType([2, 2], f32)], (b, [x, y]) => {
      b.returnOp([b.matmul(x, y).getResult(0)]);
    });
    const primFunc = lowerGraphToPrimFunc(func);
    const compiled = new BackendPipeline(CPUTarget()).compile(primFunc);
    const mod = new RuntimeModule('e2e');
    mod.addCompiledKernel(compiled);
    const A = RuntimeTensor.fromArray([1,2,3,4,5,6], [2,3]);
    const B = RuntimeTensor.fromArray([7,8,9,10,11,12], [3,2]);
    const C = RuntimeTensor.zeros([2,2]);
    mod.run('e2e_dot', A, B, C);
    assert.equal(C.get([0,0]), 1*7 + 2*9 + 3*11);
    assert.equal(C.get([0,1]), 1*8 + 2*10 + 3*12);
    assert.equal(C.get([1,0]), 4*7 + 5*9 + 6*11);
    assert.equal(C.get([1,1]), 4*8 + 5*10 + 6*12);
  });

  it('reduce sum', () => {
    const func = buildFunction('e2e_reduce', [new TensorType([4, 3], f32)], [new TensorType([4], f32)], (b, [x]) => {
      b.returnOp([b.reduce(x, b.scalarConstant(0, f32).getResult(0), [1], 'sum').getResult(0)]);
    });
    const primFunc = lowerGraphToPrimFunc(func);
    const compiled = new BackendPipeline(CPUTarget()).compile(primFunc);
    const mod = new RuntimeModule('e2e');
    mod.addCompiledKernel(compiled);
    const A = RuntimeTensor.fromArray([1,2,3, 4,5,6, 7,8,9, 10,11,12], [4,3]);
    const C = RuntimeTensor.zeros([4]);
    mod.run('e2e_reduce', A, C);
    assert.equal(C.data[0], 6);
    assert.equal(C.data[1], 15);
    assert.equal(C.data[2], 24);
    assert.equal(C.data[3], 33);
  });
});
