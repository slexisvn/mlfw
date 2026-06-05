import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

function approxEqual(a, b, eps = 1e-4) {
  return Math.abs(a - b) < eps;
}

describe('E2E pipeline: epilogue fusion enabled', () => {
  it('dot+bias through full pipeline', () => {
    const func = buildFunction('gemm_bias',
      [new TensorType([2, 3], f32), new TensorType([3, 4], f32), new TensorType([4], f32)],
      [new TensorType([2, 4], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 4], [1]);
        b.returnOp([b.add(dot.getResult(0), bcast.getResult(0)).getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: true }));
    const X = RuntimeTensor.fromArray([1,0,0, 0,1,0], [2, 3]);
    const W = RuntimeTensor.fromArray([1,2,3,4, 5,6,7,8, 9,10,11,12], [3, 4]);
    const bias = RuntimeTensor.fromArray([100, 200, 300, 400], [4]);
    const out = RuntimeTensor.zeros([2, 4]);
    compiled.run('gemm_bias', X, W, bias, out);
    assert.equal(out.data[0], 101);
    assert.equal(out.data[1], 202);
    assert.equal(out.data[4], 105);
    assert.equal(out.data[5], 206);
  });

  it('dot+bias+relu through full pipeline', () => {
    const func = buildFunction('gemm_bias_relu',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32), new TensorType([2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 2], [1]);
        const biased = b.add(dot.getResult(0), bcast.getResult(0));
        b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: true }));
    const X = RuntimeTensor.fromArray([1,0,0, 0,1,0], [2, 3]);
    const W = RuntimeTensor.fromArray([-10, 5, 3, -2, 0, 0], [3, 2]);
    const bias = RuntimeTensor.fromArray([1, 1], [2]);
    const out = RuntimeTensor.zeros([2, 2]);
    compiled.run('gemm_bias_relu', X, W, bias, out);
    assert.equal(out.data[0], 0);
    assert.equal(out.data[1], 6);
    assert.equal(out.data[2], 4);
    assert.equal(out.data[3], 0);
  });

  it('epilogue fusion disabled produces same results', () => {
    const func = buildFunction('no_epi',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32), new TensorType([2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 2], [1]);
        b.returnOp([b.add(dot.getResult(0), bcast.getResult(0)).getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const X = RuntimeTensor.fromArray([1,0,0, 0,1,0], [2, 3]);
    const W = RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6], [3, 2]);
    const bias = RuntimeTensor.fromArray([10, 20], [2]);
    const out = RuntimeTensor.zeros([2, 2]);
    compiled.run('no_epi', X, W, bias, out);
    assert.equal(out.data[0], 11);
    assert.equal(out.data[1], 22);
    assert.equal(out.data[2], 13);
    assert.equal(out.data[3], 24);
  });

  it('epilogue vs no-epilogue produce identical results', () => {
    function buildGemmBias() {
      return buildFunction('gb',
        [new TensorType([4, 8], f32), new TensorType([8, 6], f32), new TensorType([6], f32)],
        [new TensorType([4, 6], f32)],
        (b, [x, w, bias]) => {
          const dot = b.matmul(x, w);
          const bcast = b.broadcast(bias, [4, 6], [1]);
          b.returnOp([b.add(dot.getResult(0), bcast.getResult(0)).getResult(0)]);
        }
      );
    }

    const withEpi = compileGraph(buildGemmBias(), CPUTarget({ enableEpilogueFusion: true }));
    const withoutEpi = compileGraph(buildGemmBias(), CPUTarget({ enableEpilogueFusion: false }));

    const X = RuntimeTensor.fromArray(Array.from({ length: 32 }, (_, i) => (i % 7) * 0.1), [4, 8]);
    const W = RuntimeTensor.fromArray(Array.from({ length: 48 }, (_, i) => (i % 5) * 0.2 - 0.4), [8, 6]);
    const bias = RuntimeTensor.fromArray([0.1, -0.2, 0.3, -0.1, 0.5, -0.3], [6]);

    const out1 = RuntimeTensor.zeros([4, 6]);
    const out2 = RuntimeTensor.zeros([4, 6]);
    withEpi.run('gb', X, W, bias, out1);
    withoutEpi.run('gb', X, W, bias, out2);

    for (let i = 0; i < 24; i++) {
      assert.ok(approxEqual(out1.data[i], out2.data[i]),
        `mismatch at [${i}]: epilogue=${out1.data[i]} vs standard=${out2.data[i]}`);
    }
  });

  it('GPU codegen produces valid source with epilogue', () => {
    const func = buildFunction('gpu_epi',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32), new TensorType([2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 2], [1]);
        b.returnOp([b.add(dot.getResult(0), bcast.getResult(0)).getResult(0)]);
      }
    );

    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: true }));
    const source = compiled.getSource('gpu_epi');
    assert.ok(source.includes('__global__'));
    assert.ok(source.includes('+'), 'should contain bias add');
  });
});
