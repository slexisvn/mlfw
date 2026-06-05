import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';

import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';

import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';

import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;
const i32 = ScalarType.I32;
const bool = ScalarType.BOOL;

function approxEqual(a, b, eps = 1e-5) {
  return Math.abs(a - b) < eps;
}

describe('E2E: compare', () => {
  for (const [dir, jsFn] of [['eq', (a,b) => a === b], ['ne', (a,b) => a !== b], ['lt', (a,b) => a < b], ['le', (a,b) => a <= b], ['gt', (a,b) => a > b], ['ge', (a,b) => a >= b]]) {
    it(`compare direction=${dir}`, () => {
      const f32_4 = new TensorType([4], f32);
      const bool_4 = new TensorType([4], bool);
      const func = buildFunction(`cmp_${dir}`, [f32_4, f32_4], [bool_4], (b, [x, y]) => {
        b.returnOp([b.compare(x, y, dir).getResult(0)]);
      });
      const compiled = compileGraph(func, CPUTarget());
      const X = RuntimeTensor.fromArray([1, 2, 3, 4], [4]);
      const Y = RuntimeTensor.fromArray([2, 2, 2, 2], [4]);
      const out = RuntimeTensor.zeros([4]);
      compiled.run(`cmp_${dir}`, X, Y, out);
      for (let i = 0; i < 4; i++) {
        const expected = jsFn(X.data[i], Y.data[i]) ? 1 : 0;
        assert.equal(out.data[i], expected, `${dir}(${X.data[i]}, ${Y.data[i]}) expected ${expected}, got ${out.data[i]}`);
      }
    });
  }
});

describe('E2E: select', () => {
  it('selects between two tensors based on predicate', () => {
    const f32_4 = new TensorType([4], f32);
    const bool_4 = new TensorType([4], bool);
    const func = buildFunction('sel', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      const pred = b.compare(x, y, 'gt');
      const result = b.select(pred.getResult(0), x, y);
      b.returnOp([result.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1, 5, 3, 7], [4]);
    const Y = RuntimeTensor.fromArray([2, 2, 4, 4], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('sel', X, Y, out);
    assert.deepEqual([...out.data], [2, 5, 4, 7]);
  });
});

describe('E2E: clamp', () => {
  it('clamps values between lo and hi', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('clp', [f32_4, f32_4, f32_4], [f32_4], (b, [lo, x, hi]) => {
      b.returnOp([b.clamp(lo, x, hi).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const lo = RuntimeTensor.fromArray([0, 0, 0, 0], [4]);
    const X = RuntimeTensor.fromArray([-1, 0.5, 5, 1], [4]);
    const hi = RuntimeTensor.fromArray([1, 1, 1, 1], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('clp', lo, X, hi, out);
    assert.equal(out.data[0], 0);
    assert.ok(approxEqual(out.data[1], 0.5));
    assert.equal(out.data[2], 1);
    assert.equal(out.data[3], 1);
  });
});

describe('E2E: mean reduce', () => {
  it('computes row-wise mean', () => {
    const func = buildFunction('rowmean',
      [new TensorType([2, 4], f32)],
      [new TensorType([2], f32)],
      (b, [x]) => {
        const init = b.scalarConstant(0, f32);
        b.returnOp([b.reduce(x, init.getResult(0), [1], 'mean').getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1, 2, 3, 4, 10, 20, 30, 40], [2, 4]);
    const out = RuntimeTensor.zeros([2]);
    compiled.run('rowmean', X, out);
    assert.ok(approxEqual(out.data[0], 2.5));
    assert.ok(approxEqual(out.data[1], 25.0));
  });
});

describe('E2E: broadcast', () => {
  it('broadcasts scalar to vector and adds', () => {
    const f32_4 = new TensorType([4], f32);
    const scalar = new TensorType([], f32);
    const func = buildFunction('bcast_add', [f32_4, scalar], [f32_4], (b, [x, s]) => {
      const bcast = b.broadcast(s, [4], []);
      const result = b.add(x, bcast.getResult(0));
      b.returnOp([result.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1, 2, 3, 4], [4]);
    const S = RuntimeTensor.fromArray([10], []);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('bcast_add', X, S, out);
    assert.deepEqual([...out.data], [11, 12, 13, 14]);
  });

  it('broadcasts vector to matrix', () => {
    const func = buildFunction('bcast_mat',
      [new TensorType([3], f32)],
      [new TensorType([2, 3], f32)],
      (b, [x]) => {
        const bcast = b.broadcast(x, [2, 3], [1]);
        b.returnOp([bcast.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([10, 20, 30], [3]);
    const out = RuntimeTensor.zeros([2, 3]);
    compiled.run('bcast_mat', X, out);
    assert.deepEqual([...out.data], [10, 20, 30, 10, 20, 30]);
  });
});

describe('E2E: compare+select fused', () => {
  it('fused compare->select produces correct results', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('fused_cmp_sel', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      const cmp = b.compare(x, y, 'gt');
      const sel = b.select(cmp.getResult(0), x, y);
      b.returnOp([sel.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([1, 5, 3, 7], [4]);
    const Y = RuntimeTensor.fromArray([2, 2, 4, 4], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('fused_cmp_sel', X, Y, out);
    assert.deepEqual([...out.data], [2, 5, 4, 7]);
  });
});

describe('E2E: GPU codegen for compare', () => {
  it('produces valid CUDA compare operators', async () => {
    const { GPUTarget } = await import('../../../src/backend/target.js');
    const f32_4 = new TensorType([4], f32);
    const bool_4 = new TensorType([4], bool);
    const func = buildFunction('gpu_cmp', [f32_4, f32_4], [bool_4], (b, [x, y]) => {
      b.returnOp([b.compare(x, y, 'gt').getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget());
    const source = compiled.getSource('gpu_cmp');
    assert.ok(source.includes('>'), `CUDA source should contain '>' operator, got: ${source.substring(0, 200)}`);
    assert.ok(!source.includes(' gt '), `CUDA source should NOT contain raw 'gt' string as operator`);
  });
});

describe('E2E: convert (dtype cast)', () => {
  it('f32 -> i32 truncation', () => {
    const f32_4 = new TensorType([4], f32);
    const i32_4 = new TensorType([4], i32);
    const func = buildFunction('f2i', [f32_4], [i32_4], (b, [x]) => {
      b.returnOp([b.convert(x, i32).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1.9, -2.7, 0.0, 3.1], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('f2i', X, out);
    assert.equal(out.data[0], 1);
    assert.equal(out.data[1], -2);
    assert.equal(out.data[2], 0);
    assert.equal(out.data[3], 3);
  });
});

describe('E2E: pow', () => {
  it('element-wise power', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('power', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      b.returnOp([b.pow(x, y).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([2, 3, 4, 5], [4]);
    const Y = RuntimeTensor.fromArray([3, 2, 0.5, 1], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('power', X, Y, out);
    assert.ok(approxEqual(out.data[0], 8));
    assert.ok(approxEqual(out.data[1], 9));
    assert.ok(approxEqual(out.data[2], 2));
    assert.ok(approxEqual(out.data[3], 5));
  });
});

describe('E2E: rem', () => {
  it('element-wise remainder', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('remainder', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      b.returnOp([b.rem(x, y).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([7, 10, 5, 3], [4]);
    const Y = RuntimeTensor.fromArray([3, 4, 2, 7], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('remainder', X, Y, out);
    assert.ok(approxEqual(out.data[0], 1));
    assert.ok(approxEqual(out.data[1], 2));
    assert.ok(approxEqual(out.data[2], 1));
    assert.ok(approxEqual(out.data[3], 3));
  });
});

describe('E2E: sin / cos', () => {
  it('sin', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('sinf', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.sin(x).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([0, Math.PI / 2, Math.PI, 3 * Math.PI / 2], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('sinf', X, out);
    assert.ok(approxEqual(out.data[0], 0));
    assert.ok(approxEqual(out.data[1], 1));
    assert.ok(approxEqual(out.data[2], 0, 1e-3));
    assert.ok(approxEqual(out.data[3], -1));
  });

  it('cos', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('cosf', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.cos(x).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([0, Math.PI / 2, Math.PI, 2 * Math.PI], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('cosf', X, out);
    assert.ok(approxEqual(out.data[0], 1));
    assert.ok(approxEqual(out.data[1], 0, 1e-3));
    assert.ok(approxEqual(out.data[2], -1));
    assert.ok(approxEqual(out.data[3], 1));
  });
});

describe('E2E: round', () => {
  it('rounds to nearest integer', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('roundf', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b._inferAndBuild('round', [x]).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1.4, 1.5, 2.6, -0.7], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('roundf', X, out);
    assert.equal(out.data[0], Math.round(1.4));
    assert.equal(out.data[1], Math.round(1.5));
    assert.equal(out.data[2], Math.round(2.6));
    assert.equal(out.data[3], Math.round(-0.7));
  });
});

describe('E2E: sign', () => {
  it('returns sign of elements', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('signf', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.sign(x).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([5, -3, 0, -0.001], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('signf', X, out);
    assert.equal(out.data[0], Math.sign(5));
    assert.equal(out.data[1], Math.sign(-3));
    assert.equal(out.data[2], Math.sign(0));
    assert.equal(out.data[3], Math.sign(-0.001));
  });
});

describe('E2E pattern: where(x > 0, x, 0)', () => {
  it('relu via compare+select', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('where_relu', [f32_4], [f32_4], (b, [x]) => {
      const zero = b.scalarConstant(0, f32);
      const bcastZero = b.broadcast(zero.getResult(0), [4], []);
      const pred = b.compare(x, bcastZero.getResult(0), 'gt');
      const result = b.select(pred.getResult(0), x, bcastZero.getResult(0));
      b.returnOp([result.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([-2, 3, 0, -0.5], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('where_relu', X, out);
    assert.deepEqual([...out.data], [0, 3, 0, 0]);
  });

  it('where(x > 0, x, 0) fused', () => {
    const f32_8 = new TensorType([8], f32);
    const func = buildFunction('where_relu_fused', [f32_8], [f32_8], (b, [x]) => {
      const zero = b.scalarConstant(0, f32);
      const bcastZero = b.broadcast(zero.getResult(0), [8], []);
      const pred = b.compare(x, bcastZero.getResult(0), 'gt');
      const result = b.select(pred.getResult(0), x, bcastZero.getResult(0));
      b.returnOp([result.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([-1, 2, -3, 4, 0, 5, -6, 7], [8]);
    const out = RuntimeTensor.zeros([8]);
    compiled.run('where_relu_fused', X, out);
    assert.deepEqual([...out.data], [0, 2, 0, 4, 0, 5, 0, 7]);
  });
});

describe('E2E pattern: clamp(x, 0, 6) — relu6', () => {
  it('scalar broadcast clamp', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('relu6', [f32_4], [f32_4], (b, [x]) => {
      const lo = b.broadcast(b.scalarConstant(0, f32).getResult(0), [4], []);
      const hi = b.broadcast(b.scalarConstant(6, f32).getResult(0), [4], []);
      b.returnOp([b.clamp(lo.getResult(0), x, hi.getResult(0)).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([-2, 3, 7, 6], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('relu6', X, out);
    assert.deepEqual([...out.data], [0, 3, 6, 6]);
  });
});

describe('E2E pattern: mean(x, axis)', () => {
  it('column-wise mean on 3x4', () => {
    const func = buildFunction('colmean',
      [new TensorType([3, 4], f32)],
      [new TensorType([4], f32)],
      (b, [x]) => {
        const init = b.scalarConstant(0, f32);
        b.returnOp([b.reduce(x, init.getResult(0), [0], 'mean').getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12
    ], [3, 4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('colmean', X, out);
    assert.ok(approxEqual(out.data[0], 5));
    assert.ok(approxEqual(out.data[1], 6));
    assert.ok(approxEqual(out.data[2], 7));
    assert.ok(approxEqual(out.data[3], 8));
  });

  it('full reduce mean (single element output)', () => {
    const func = buildFunction('fullmean',
      [new TensorType([2, 3], f32)],
      [new TensorType([], f32)],
      (b, [x]) => {
        const init = b.scalarConstant(0, f32);
        b.returnOp([b.reduce(x, init.getResult(0), [0, 1], 'mean').getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const out = RuntimeTensor.zeros([]);
    compiled.run('fullmean', X, out);
    assert.ok(approxEqual(out.data[0], 3.5));
  });
});

describe('E2E pattern: layernorm primitive', () => {
  it('layernorm normalizes along last axis', () => {
    const func = buildFunction('ln',
      [new TensorType([2, 4], f32), new TensorType([4], f32), new TensorType([4], f32)],
      [new TensorType([2, 4], f32)],
      (b, [x, gamma, beta]) => {
        const result = b.layernorm(x, gamma, beta, -1, 1e-5);
        b.returnOp([result.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1, 2, 3, 4, 2, 4, 6, 8], [2, 4]);
    const gamma = RuntimeTensor.fromArray([1, 1, 1, 1], [4]);
    const beta = RuntimeTensor.fromArray([0, 0, 0, 0], [4]);
    const out = RuntimeTensor.zeros([2, 4]);
    compiled.run('ln', X, gamma, beta, out);
    for (let row = 0; row < 2; row++) {
      let sum = 0;
      for (let j = 0; j < 4; j++) sum += out.data[row * 4 + j];
      assert.ok(approxEqual(sum, 0, 1e-3), `row ${row} mean should be ~0, got ${sum / 4}`);
    }
  });
});

describe('E2E pattern: broadcast bias add', () => {
  it('2D input + 1D bias broadcast', () => {
    const func = buildFunction('bias_add',
      [new TensorType([3, 4], f32), new TensorType([4], f32)],
      [new TensorType([3, 4], f32)],
      (b, [x, bias]) => {
        const bcast = b.broadcast(bias, [3, 4], [1]);
        b.returnOp([b.add(x, bcast.getResult(0)).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1,2,3,4, 5,6,7,8, 9,10,11,12], [3, 4]);
    const bias = RuntimeTensor.fromArray([10, 20, 30, 40], [4]);
    const out = RuntimeTensor.zeros([3, 4]);
    compiled.run('bias_add', X, bias, out);
    assert.deepEqual([...out.data], [11,22,33,44, 15,26,37,48, 19,30,41,52]);
  });

  it('3D input + 1D bias broadcast on last dim', () => {
    const func = buildFunction('bias_add_3d',
      [new TensorType([2, 2, 3], f32), new TensorType([3], f32)],
      [new TensorType([2, 2, 3], f32)],
      (b, [x, bias]) => {
        const bcast = b.broadcast(bias, [2, 2, 3], [2]);
        b.returnOp([b.add(x, bcast.getResult(0)).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1,2,3, 4,5,6, 7,8,9, 10,11,12], [2, 2, 3]);
    const bias = RuntimeTensor.fromArray([100, 200, 300], [3]);
    const out = RuntimeTensor.zeros([2, 2, 3]);
    compiled.run('bias_add_3d', X, bias, out);
    assert.deepEqual([...out.data], [101,202,303, 104,205,306, 107,208,309, 110,211,312]);
  });
});

describe('E2E: broadcast_in_dim with size-1 dimension', () => {
  it('broadcasts [1, 4] -> [3, 4]', () => {
    const func = buildFunction('bcast_size1',
      [new TensorType([1, 4], f32)],
      [new TensorType([3, 4], f32)],
      (b, [x]) => {
        const bcast = b.broadcast(x, [3, 4], [0, 1]);
        b.returnOp([bcast.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([10, 20, 30, 40], [1, 4]);
    const out = RuntimeTensor.zeros([3, 4]);
    compiled.run('bcast_size1', X, out);
    assert.deepEqual([...out.data], [10,20,30,40, 10,20,30,40, 10,20,30,40]);
  });
});

describe('E2E: GPU codegen for new ops', () => {
  it('GPU pow produces powf call', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('gpu_pow', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      b.returnOp([b.pow(x, y).getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget());
    const source = compiled.getSource('gpu_pow');
    assert.ok(source.includes('powf'), `CUDA source should contain powf, got: ${source.substring(0, 300)}`);
  });

  it('GPU sin/cos produce sinf/cosf calls', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('gpu_trig', [f32_4], [f32_4], (b, [x]) => {
      const s = b.sin(x);
      b.returnOp([b.cos(s.getResult(0)).getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget());
    const source = compiled.getSource('gpu_trig');
    assert.ok(source.includes('sinf'), `CUDA source should contain sinf`);
    assert.ok(source.includes('cosf'), `CUDA source should contain cosf`);
  });

  it('GPU convert produces cast', () => {
    const f32_4 = new TensorType([4], f32);
    const i32_4 = new TensorType([4], i32);
    const func = buildFunction('gpu_cast', [f32_4], [i32_4], (b, [x]) => {
      b.returnOp([b.convert(x, i32).getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget());
    const source = compiled.getSource('gpu_cast');
    assert.ok(source.includes('int'), `CUDA source should contain int cast, got: ${source.substring(0, 300)}`);
  });
});

describe('Verifier: reshape numel check', () => {
  it('rejects numel mismatch', () => {
    assert.throws(() => {
      const func = buildFunction('bad_reshape',
        [new TensorType([3, 4], f32)],
        [new TensorType([5, 3], f32)],
        (b, [x]) => {
          b.returnOp([b.reshape(x, [5, 3]).getResult(0)]);
        }
      );
      compileGraph(func, CPUTarget());
    }, /numel mismatch/);
  });
});

describe('Verifier: dot contracting dim check', () => {
  it('rejects mismatched contracting dim sizes', () => {
    assert.throws(() => {
      const func = buildFunction('bad_dot',
        [new TensorType([2, 3], f32), new TensorType([5, 2], f32)],
        [new TensorType([2, 2], f32)],
        (b, [x, y]) => {
          b.returnOp([b.dot(x, y, [1], [0]).getResult(0)]);
        }
      );
      compileGraph(func, CPUTarget());
    }, /contracting dim size mismatch/);
  });
});

describe('Lowering: unsupported op throws', () => {
  it('rejects op without lowering rule', () => {
    assert.throws(() => {
      const func = buildFunction('bad_custom',
        [new TensorType([10], f32)],
        [new TensorType([10], f32)],
        (b, [x]) => {
          b.returnOp([b.customCall('unknown_lib_fn', [x], [new TensorType([10], f32)]).getResult(0)]);
        }
      );
      compileGraph(func, CPUTarget());
    }, /No lowering rule/);
  });
});