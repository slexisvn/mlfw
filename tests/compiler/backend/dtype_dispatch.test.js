import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { RuntimeMemoryManager } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { dtypeInfo, cType, cPtrType, jsTypedArray, cMathFunc, libraryFunc } from '../../../src/compiler/backend/dtype_map.js';
import { createCPULibrarySelector, createGPULibrarySelector } from '../../../src/compiler/backend/library_selector.js';

describe('dtype_map', () => {
  it('f32 maps to Float32Array and float', () => {
    assert.equal(jsTypedArray('f32'), 'Float32Array');
    assert.equal(cType('f32'), 'float');
    assert.equal(cPtrType('f32'), 'float*');
  });

  it('f64 maps to Float64Array and double', () => {
    assert.equal(jsTypedArray('f64'), 'Float64Array');
    assert.equal(cType('f64'), 'double');
    assert.equal(cPtrType('f64'), 'double*');
  });

  it('i32 maps to Int32Array and int', () => {
    assert.equal(jsTypedArray('i32'), 'Int32Array');
    assert.equal(cType('i32'), 'int');
  });

  it('bool maps to Uint8Array and bool', () => {
    assert.equal(jsTypedArray('bool'), 'Uint8Array');
    assert.equal(cType('bool'), 'bool');
  });

  it('cMathFunc dispatches by dtype', () => {
    assert.equal(cMathFunc('exp', 'f32'), 'expf');
    assert.equal(cMathFunc('exp', 'f64'), 'exp');
    assert.equal(cMathFunc('sin', 'f32'), 'sinf');
    assert.equal(cMathFunc('sin', 'f64'), 'sin');
    assert.equal(cMathFunc('max', 'f32'), 'fmaxf');
  });

  it('libraryFunc dispatches by dtype', () => {
    assert.equal(libraryFunc('dot', 'blas', 'f32'), 'sgemm');
    assert.equal(libraryFunc('dot', 'blas', 'f64'), 'dgemm');
    assert.equal(libraryFunc('dot', 'cublas', 'f16'), 'cublasHgemm');
    assert.equal(libraryFunc('dot', 'cublas', 'f32'), 'cublasSgemm');
    assert.equal(libraryFunc('conv', 'cudnn', 'f32'), 'cudnnConvolutionForward');
    assert.equal(libraryFunc('dot', 'blas', 'i32'), null);
  });

  it('dtypeInfo returns full entry', () => {
    const info = dtypeInfo('f32');
    assert.equal(info.bytes, 4);
    assert.equal(info.isFloat, true);
    assert.equal(info.isInt, false);
  });
});

describe('CPU codegen dtype-aware temps', () => {
  it('i32 output uses Int32Array for temps', () => {
    const func = buildFunction('i32_add',
      [new TensorType([4], ScalarType.I32), new TensorType([4], ScalarType.I32)],
      [new TensorType([4], ScalarType.I32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false, enableEpilogueFusion: false });
    const source = compiled.getSource('i32_add');
    assert.ok(!source.includes('Float32Array') || source.includes('Int32Array'),
      'should not hardcode Float32Array for i32 buffers');
  });

  it('f64 computation compiles', () => {
    const func = buildFunction('f64_exp',
      [new TensorType([4], ScalarType.F64)],
      [new TensorType([4], ScalarType.F64)],
      (b, [x]) => {
        b.returnOp([b.exp(x).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false, enableEpilogueFusion: false });
    const source = compiled.getSource('f64_exp');
    assert.ok(source.includes('Math.exp'));
  });
});

describe('GPU codegen dtype-aware params', () => {
  it('f64 params use double*', () => {
    const func = buildFunction('gpu_f64',
      [new TensorType([4], ScalarType.F64), new TensorType([4], ScalarType.F64)],
      [new TensorType([4], ScalarType.F64)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, GPUTarget(), { enableEpilogueFusion: false });
    const source = compiled.getSource('gpu_f64');
    assert.ok(source.includes('double*'), `should use double* for f64, got:\n${source.substring(0, 200)}`);
  });

  it('i32 params use int*', () => {
    const func = buildFunction('gpu_i32',
      [new TensorType([4], ScalarType.I32), new TensorType([4], ScalarType.I32)],
      [new TensorType([4], ScalarType.I32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, GPUTarget(), { enableEpilogueFusion: false });
    const source = compiled.getSource('gpu_i32');
    assert.ok(source.includes('int*'), `should use int* for i32, got:\n${source.substring(0, 200)}`);
  });
});

describe('RuntimeTensor dtype-aware allocation', () => {
  it('zeros f64 creates Float64Array', () => {
    const t = RuntimeTensor.zeros([4], 'f64');
    assert.ok(t.data instanceof Float64Array);
  });

  it('zeros i32 creates Int32Array', () => {
    const t = RuntimeTensor.zeros([4], 'i32');
    assert.ok(t.data instanceof Int32Array);
  });

  it('zeros bool creates Uint8Array', () => {
    const t = RuntimeTensor.zeros([4], 'bool');
    assert.ok(t.data instanceof Uint8Array);
  });

  it('fromArray f64 creates Float64Array', () => {
    const t = RuntimeTensor.fromArray([1.0, 2.0], [2], 'f64');
    assert.ok(t.data instanceof Float64Array);
    assert.equal(t.data[0], 1.0);
  });
});

describe('RuntimeMemoryManager dtype-aware pool', () => {
  it('allocates correct typed array for dtype', () => {
    const mgr = new RuntimeMemoryManager();
    const f32buf = mgr.allocate(16, 'f32');
    assert.ok(f32buf instanceof Float32Array);
    const i32buf = mgr.allocate(16, 'i32');
    assert.ok(i32buf instanceof Int32Array);
    const f64buf = mgr.allocate(16, 'f64');
    assert.ok(f64buf instanceof Float64Array);
  });

  it('reuses pooled buffers after release', () => {
    const mgr = new RuntimeMemoryManager();
    const buf1 = mgr.allocate(32, 'f32');
    mgr.release(buf1, 32, 'f32');
    const buf2 = mgr.allocate(32, 'f32');
    assert.equal(buf1, buf2);
  });

  it('does not mix dtype pools', () => {
    const mgr = new RuntimeMemoryManager();
    const f32buf = mgr.allocate(32, 'f32');
    mgr.release(f32buf, 32, 'f32');
    const i32buf = mgr.allocate(32, 'i32');
    assert.notEqual(f32buf, i32buf);
    assert.ok(i32buf instanceof Int32Array);
  });
});

describe('LibrarySelector dtype dispatch', () => {
  it('CPU selector picks sgemm for f32 dot', () => {
    const sel = createCPULibrarySelector(CPUTarget());
    const call = sel.select('dot', [64, 64], 'f32');
    assert.ok(call);
    assert.equal(call.functionName, 'sgemm');
  });

  it('CPU selector picks dgemm for f64 dot', () => {
    const sel = createCPULibrarySelector(CPUTarget());
    const call = sel.select('dot', [64, 64], 'f64');
    assert.ok(call);
    assert.equal(call.functionName, 'dgemm');
  });

  it('CPU selector returns null for i32 dot (not supported)', () => {
    const sel = createCPULibrarySelector(CPUTarget());
    const call = sel.select('dot', [64, 64], 'i32');
    assert.equal(call, null);
  });

  it('GPU selector picks cublasHgemm for f16 dot', () => {
    const sel = createGPULibrarySelector(GPUTarget());
    const call = sel.select('dot', [64, 64], 'f16');
    assert.ok(call);
    assert.equal(call.functionName, 'cublasHgemm');
  });

  it('GPU selector respects minElements constraint', () => {
    const sel = createCPULibrarySelector(CPUTarget());
    const call = sel.select('dot', [2, 2], 'f32');
    assert.equal(call, null);
  });
});
