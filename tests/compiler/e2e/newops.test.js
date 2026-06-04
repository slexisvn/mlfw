import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;
const i32 = ScalarType.I32;

function close(a, b, eps = 1e-3) { return Math.abs(a - b) < eps; }

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), {
    enableFusion: false, enableEpilogueFusion: false, ...opts
  });
}

function T(shape, dtype = f32) { return new TensorType(shape, dtype); }

describe('Slice', () => {
  it('1D slice [2:7]', () => {
    const func = buildFunction('s1d', [T([10])], [T([5])], (b, [x]) => {
      b.returnOp([b.slice(x, [2], [7]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: 10 }, (_, i) => i * 10), [10]);
    const out = RuntimeTensor.zeros([5]);
    compiled.run('s1d', X, out);
    assert.deepEqual([...out.data], [20, 30, 40, 50, 60]);
  });

  it('2D slice with strides', () => {
    const func = buildFunction('s2d', [T([4, 6])], [T([2, 3])], (b, [x]) => {
      b.returnOp([b.slice(x, [0, 0], [4, 6], [2, 2]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: 24 }, (_, i) => i), [4, 6]);
    const out = RuntimeTensor.zeros([2, 3]);
    compiled.run('s2d', X, out);
    assert.equal(out.data[0], 0);
    assert.equal(out.data[1], 2);
    assert.equal(out.data[2], 4);
    assert.equal(out.data[3], 12);
    assert.equal(out.data[4], 14);
    assert.equal(out.data[5], 16);
  });
});

describe('Pad', () => {
  it('1D zero padding', () => {
    const func = buildFunction('pad1d', [T([3]), T([])], [T([7])], (b, [x, pv]) => {
      b.returnOp([b.pad(x, pv, [2], [2]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([10, 20, 30], [3]);
    const PV = RuntimeTensor.fromArray([0], []);
    const out = RuntimeTensor.zeros([7]);
    compiled.run('pad1d', X, PV, out);
    assert.deepEqual([...out.data], [0, 0, 10, 20, 30, 0, 0]);
  });

  it('2D asymmetric padding', () => {
    const func = buildFunction('pad2d', [T([2, 3]), T([])], [T([4, 5])], (b, [x, pv]) => {
      b.returnOp([b.pad(x, pv, [1, 0], [1, 2]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([1,2,3, 4,5,6], [2, 3]);
    const PV = RuntimeTensor.fromArray([-1], []);
    const out = RuntimeTensor.zeros([4, 5]);
    compiled.run('pad2d', X, PV, out);
    assert.equal(out.data[0 * 5 + 0], -1);
    assert.equal(out.data[1 * 5 + 0], 1);
    assert.equal(out.data[1 * 5 + 2], 3);
    assert.equal(out.data[1 * 5 + 3], -1);
    assert.equal(out.data[3 * 5 + 0], -1);
  });
});

describe('Concat', () => {
  it('concat two vectors', () => {
    const func = buildFunction('cat1d', [T([3]), T([4])], [T([7])], (b, [x, y]) => {
      b.returnOp([b.concat([x, y], 0).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([1, 2, 3], [3]);
    const Y = RuntimeTensor.fromArray([4, 5, 6, 7], [4]);
    const out = RuntimeTensor.zeros([7]);
    compiled.run('cat1d', X, Y, out);
    assert.deepEqual([...out.data], [1, 2, 3, 4, 5, 6, 7]);
  });

  it('concat along dim 1', () => {
    const func = buildFunction('cat2d', [T([2, 3]), T([2, 2])], [T([2, 5])], (b, [x, y]) => {
      b.returnOp([b.concat([x, y], 1).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([1,2,3, 4,5,6], [2, 3]);
    const Y = RuntimeTensor.fromArray([7,8, 9,10], [2, 2]);
    const out = RuntimeTensor.zeros([2, 5]);
    compiled.run('cat2d', X, Y, out);
    assert.deepEqual([...out.data], [1,2,3,7,8, 4,5,6,9,10]);
  });

  it('concat three tensors', () => {
    const func = buildFunction('cat3', [T([2]), T([3]), T([1])], [T([6])], (b, [a, bb, c]) => {
      b.returnOp([b.concat([a, bb, c], 0).getResult(0)]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([6]);
    compiled.run('cat3', RuntimeTensor.fromArray([1,2], [2]), RuntimeTensor.fromArray([3,4,5], [3]), RuntimeTensor.fromArray([6], [1]), out);
    assert.deepEqual([...out.data], [1,2,3,4,5,6]);
  });
});

describe('Iota', () => {
  it('1D iota', () => {
    const func = buildFunction('iota1d', [], [T([5])], (b, []) => {
      b.returnOp([b.iota(0, T([5])).getResult(0)]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([5]);
    compiled.run('iota1d', out);
    assert.deepEqual([...out.data], [0, 1, 2, 3, 4]);
  });

  it('2D iota along dim 1', () => {
    const func = buildFunction('iota2d', [], [T([3, 4])], (b, []) => {
      b.returnOp([b.iota(1, T([3, 4])).getResult(0)]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([3, 4]);
    compiled.run('iota2d', out);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 4; c++)
        assert.equal(out.data[r * 4 + c], c, `iota[${r},${c}]`);
  });
});

describe('Conv2D', () => {
  it('1x1 conv (pointwise)', () => {
    const func = buildFunction('conv1x1', [T([1, 2, 3, 3]), T([4, 2, 1, 1])], [T([1, 4, 3, 3])], (b, [x, k]) => {
      b.returnOp([b.conv(x, k, [1, 1], [[0,0],[0,0]]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: 18 }, (_, i) => i), [1, 2, 3, 3]);
    const K = RuntimeTensor.fromArray(Float32Array.from({ length: 8 }, () => 1), [4, 2, 1, 1]);
    const out = RuntimeTensor.zeros([1, 4, 3, 3]);
    compiled.run('conv1x1', X, K, out);
    for (let oc = 0; oc < 4; oc++)
      for (let h = 0; h < 3; h++)
        for (let w = 0; w < 3; w++) {
          let expected = 0;
          for (let ic = 0; ic < 2; ic++) expected += X.data[ic * 9 + h * 3 + w];
          assert.ok(close(out.data[oc * 9 + h * 3 + w], expected),
            `conv1x1[${oc},${h},${w}]: ${out.data[oc * 9 + h * 3 + w]} != ${expected}`);
        }
  });

  it('3x3 conv with padding', () => {
    const func = buildFunction('conv3x3', [T([1, 1, 4, 4]), T([1, 1, 3, 3])], [T([1, 1, 4, 4])], (b, [x, k]) => {
      b.returnOp([b.conv(x, k, [1, 1], [[1,1],[1,1]]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: 16 }, () => 1), [1, 1, 4, 4]);
    const K = RuntimeTensor.fromArray(Float32Array.from({ length: 9 }, () => 1), [1, 1, 3, 3]);
    const out = RuntimeTensor.zeros([1, 1, 4, 4]);
    compiled.run('conv3x3', X, K, out);
    assert.equal(out.data[0], 4);
    assert.equal(out.data[5], 9);
    assert.equal(out.data[15], 4);
  });

  it('conv with stride 2', () => {
    const func = buildFunction('convs2', [T([1, 1, 4, 4]), T([1, 1, 3, 3])], [T([1, 1, 1, 1])], (b, [x, k]) => {
      b.returnOp([b.conv(x, k, [2, 2], [[0,0],[0,0]]).getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: 16 }, () => 1), [1, 1, 4, 4]);
    const K = RuntimeTensor.fromArray(Float32Array.from({ length: 9 }, () => 1), [1, 1, 3, 3]);
    const out = RuntimeTensor.zeros([1, 1, 1, 1]);
    compiled.run('convs2', X, K, out);
    assert.equal(out.data[0], 9);
  });
});

describe('Gather', () => {
  it('simple 1D gather', () => {
    const func = buildFunction('gather1d',
      [T([5]), T([3, 1], i32)],
      [T([3, 1])],
      (b, [data, indices]) => {
        b.returnOp([b._inferAndBuild('gather', [data, indices], {
          offset_dims: [1],
          collapsed_slice_dims: [],
          start_index_map: [0],
          slice_sizes: [1],
          index_vector_dim: 1
        }, null, [T([3, 1])]).getResult(0)]);
      }
    );
    const compiled = compile(func);
    const data = RuntimeTensor.fromArray([10, 20, 30, 40, 50], [5]);
    const indices = RuntimeTensor.fromArray(new Int32Array([1, 3, 0]), [3, 1]);
    const out = RuntimeTensor.zeros([3, 1]);
    compiled.run('gather1d', data, indices, out);
    assert.equal(out.data[0], 20);
    assert.equal(out.data[1], 40);
    assert.equal(out.data[2], 10);
  });
});

describe('Slice + Concat pipeline (split then rejoin)', () => {
  it('split vector in half then concat back', () => {
    const func = buildFunction('split_rejoin', [T([8])], [T([8])], (b, [x]) => {
      const first = b.slice(x, [0], [4]);
      const second = b.slice(x, [4], [8]);
      const rejoined = b.concat([second.getResult(0), first.getResult(0)], 0);
      b.returnOp([rejoined.getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([1,2,3,4,5,6,7,8], [8]);
    const out = RuntimeTensor.zeros([8]);
    compiled.run('split_rejoin', X, out);
    assert.deepEqual([...out.data], [5,6,7,8,1,2,3,4]);
  });
});

describe('Conv + Relu + Pool pattern', () => {
  it('conv -> relu -> global max pool', () => {
    const func = buildFunction('conv_relu_pool',
      [T([1, 1, 4, 4]), T([2, 1, 3, 3])],
      [T([1, 2])],
      (b, [x, k]) => {
        const conv = b.conv(x, k, [1, 1], [[0,0],[0,0]]);
        const act = b.relu(conv.getResult(0));
        const init = b.scalarConstant(-Infinity, f32);
        const pool = b.reduce(act.getResult(0), init.getResult(0), [2, 3], 'max');
        b.returnOp([pool.getResult(0)]);
      }
    );
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: 16 }, (_, i) => i - 8), [1, 1, 4, 4]);
    const K = RuntimeTensor.fromArray(Float32Array.from({ length: 18 }, () => 0.1), [2, 1, 3, 3]);
    const out = RuntimeTensor.zeros([1, 2]);
    compiled.run('conv_relu_pool', X, K, out);
    assert.ok(isFinite(out.data[0]) && out.data[0] >= 0);
    assert.ok(isFinite(out.data[1]) && out.data[1] >= 0);
  });
});

describe('GPU codegen for new ops', () => {
  it('conv compiles to CUDA', () => {
    const func = buildFunction('gpu_conv', [T([1, 1, 4, 4]), T([1, 1, 3, 3])], [T([1, 1, 2, 2])], (b, [x, k]) => {
      b.returnOp([b.conv(x, k, [1, 1], [[0,0],[0,0]]).getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    const source = compiled.getSource('gpu_conv');
    assert.ok(source.includes('__global__'));
  });

  it('slice + pad compiles to CUDA', () => {
    const func = buildFunction('gpu_sp', [T([10]), T([])], [T([9])], (b, [x, pv]) => {
      const sliced = b.slice(x, [1], [8]);
      const padded = b.pad(sliced.getResult(0), pv, [1], [1]);
      b.returnOp([padded.getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), { enableFusion: false, enableEpilogueFusion: false });
    assert.ok(compiled.getSource('gpu_sp').includes('__global__'));
  });
});
