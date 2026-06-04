import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction, buildModule } from '../../../src/compiler/ir/graph/builder.js';
import { GraphModule } from '../../../src/compiler/ir/graph/module.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { Compiler, CompilerConfig, compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

function approxEqual(a, b, eps = 1e-5) {
  return Math.abs(a - b) < eps;
}

describe('Compiler pipeline', () => {
  it('compiles a single function end-to-end', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('add', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    const result = compileGraph(func, CPUTarget());
    assert.ok(result.listKernels().includes('add'));
    assert.ok(result.getSource('add').includes('function add('));
  });

  it('compiles module with multiple functions', () => {
    const f32_4 = new TensorType([4], f32);
    const mod = new GraphModule('multi');
    mod.addFunction(buildFunction('f1', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0)]);
    }));
    mod.addFunction(buildFunction('f2', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.neg(x).getResult(0)]);
    }));
    const compiler = new Compiler({ target: CPUTarget() });
    const result = compiler.compile(mod);
    assert.ok(result.listKernels().includes('f1'));
    assert.ok(result.listKernels().includes('f2'));
  });

  it('requires target', () => {
    assert.throws(() => new Compiler({}), /requires a target/);
  });
});

describe('E2E: elementwise chain', () => {
  it('add + mul + exp', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('chain', [f32_4, f32_4], [f32_4], (b, [x, y]) => {
      const a = b.add(x, y);
      const m = b.mul(a.getResult(0), x);
      b.returnOp([b.exp(m.getResult(0)).getResult(0)]);
    });
    const result = compileGraph(func, CPUTarget());
    const A = RuntimeTensor.fromArray([1, 0, 0, 0], [4]);
    const B = RuntimeTensor.fromArray([0, 0, 0, 0], [4]);
    const C = RuntimeTensor.zeros([4]);
    result.run('chain', A, B, C);
    assert.ok(approxEqual(C.data[0], Math.exp(1)));
    assert.ok(approxEqual(C.data[1], Math.exp(0)));
  });
});

describe('E2E: matmul + bias + relu', () => {
  it('2x3 @ 3x2 + bias', () => {
    const func = buildFunction('matmul_bias_relu',
      [new TensorType([2, 3], f32), new TensorType([3, 2], f32), new TensorType([2], f32)],
      [new TensorType([2, 2], f32)],
      (b, [x, w, bias]) => {
        const dot = b.matmul(x, w);
        const bcast = b.broadcast(bias, [2, 2], [1]);
        const added = b.add(dot.getResult(0), bcast.getResult(0));
        const result = b.relu(added.getResult(0));
        b.returnOp([result.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([1,2,3,4,5,6], [2,3]);
    const W = RuntimeTensor.fromArray([1,0,0,1,0,0], [3,2]);
    const bias = RuntimeTensor.fromArray([-10, 5], [2]);
    const out = RuntimeTensor.zeros([2,2]);
    compiled.run('matmul_bias_relu', X, W, bias, out);
    assert.ok(out.data[0] >= 0);
  });
});

describe('E2E: softmax', () => {
  it('softmax sums to 1', () => {
    const f32_1x4 = new TensorType([1, 4], f32);
    const func = buildFunction('softmax_func', [f32_1x4], [f32_1x4], (b, [x]) => {
      const result = b.softmax(x, -1);
      b.returnOp([result.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: false });
    const X = RuntimeTensor.fromArray([1, 2, 3, 4], [1, 4]);
    const out = RuntimeTensor.zeros([1, 4]);
    compiled.run('softmax_func', X, out);
    let sum = 0;
    for (let i = 0; i < 4; i++) sum += out.data[i];
    assert.ok(approxEqual(sum, 1.0, 1e-4));
    for (let i = 0; i < 4; i++) assert.ok(out.data[i] > 0);
    assert.ok(out.data[3] > out.data[0]);
  });
});

describe('E2E: reduce sum', () => {
  it('row-wise sum', () => {
    const func = buildFunction('rowsum', [new TensorType([3, 4], f32)], [new TensorType([3], f32)], (b, [x]) => {
      const init = b.scalarConstant(0, f32);
      b.returnOp([b.reduce(x, init.getResult(0), [1], 'sum').getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([1,2,3,4, 5,6,7,8, 9,10,11,12], [3,4]);
    const out = RuntimeTensor.zeros([3]);
    compiled.run('rowsum', X, out);
    assert.equal(out.data[0], 10);
    assert.equal(out.data[1], 26);
    assert.equal(out.data[2], 42);
  });
});

describe('E2E: algebraic simplification through pipeline', () => {
  it('x + 0 * 1 optimized away', () => {
    const f32_4 = new TensorType([4], f32);
    const scalar = new TensorType([], f32);
    const func = buildFunction('identity', [f32_4], [f32_4], (b, [x]) => {
      const zero = b.scalarConstant(0, f32);
      const one = b.scalarConstant(1, f32);
      const added = b.add(x, b.broadcast(zero.getResult(0), [4], []).getResult(0));
      const mulled = b.mul(added.getResult(0), b.broadcast(one.getResult(0), [4], []).getResult(0));
      b.returnOp([mulled.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget());
    const X = RuntimeTensor.fromArray([42, 7, 13, 99], [4]);
    const out = RuntimeTensor.zeros([4]);
    compiled.run('identity', X, out);
    assert.deepEqual([...out.data], [42, 7, 13, 99]);
  });
});

describe('E2E: fusion through pipeline', () => {
  it('fused elementwise chain produces correct results', () => {
    const f32_8 = new TensorType([8], f32);
    const func = buildFunction('fused', [f32_8, f32_8], [f32_8], (b, [x, y]) => {
      const a = b.add(x, y);
      const m = b.mul(a.getResult(0), x);
      const s = b.sub(m.getResult(0), y);
      b.returnOp([s.getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableFusion: true });
    const X = RuntimeTensor.fromArray([1,2,3,4,5,6,7,8], [8]);
    const Y = RuntimeTensor.fromArray([1,1,1,1,1,1,1,1], [8]);
    const out = RuntimeTensor.zeros([8]);
    compiled.run('fused', X, Y, out);
    for (let i = 0; i < 8; i++) {
      const x = i + 1, y = 1;
      assert.equal(out.data[i], (x + y) * x - y);
    }
  });
});

describe('E2E: GPU codegen pipeline', () => {
  it('produces CUDA kernel source', () => {
    const f32_256 = new TensorType([256], f32);
    const func = buildFunction('gpu_add', [f32_256, f32_256], [f32_256], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    const compiled = compileGraph(func, GPUTarget(), { enableAutotune: false });
    const source = compiled.getSource('gpu_add');
    assert.ok(source.includes('__global__'));
  });
});

describe('E2E: autotune pipeline', () => {
  it('compiles with autotune enabled', () => {
    const f32_64 = new TensorType([64], f32);
    const func = buildFunction('autotuned', [f32_64, f32_64], [f32_64], (b, [x, y]) => {
      b.returnOp([b.add(x, y).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: { strategy: 'random', numTrials: 4, seed: 42 }
    });
    const X = RuntimeTensor.fromArray(new Array(64).fill(1), [64]);
    const Y = RuntimeTensor.fromArray(new Array(64).fill(2), [64]);
    const out = RuntimeTensor.zeros([64]);
    compiled.run('autotuned', X, Y, out);
    assert.equal(out.data[0], 3);
    assert.equal(out.data[63], 3);
  });
});

describe('E2E: diagnostics', () => {
  it('collects phase diagnostics when enabled', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('diag', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableDiagnostics: true });
    assert.ok(compiled.diagnostics);
    assert.ok(compiled.diagnostics.entries.length > 0);
    const phases = compiled.diagnostics.summary();
    assert.ok(phases.has('graphPasses'));
    assert.ok(phases.has('lowering'));
    assert.ok(phases.has('codegen'));
  });

  it('no diagnostics when disabled', () => {
    const f32_4 = new TensorType([4], f32);
    const func = buildFunction('nodiag', [f32_4], [f32_4], (b, [x]) => {
      b.returnOp([b.exp(x).getResult(0)]);
    });
    const compiled = compileGraph(func, CPUTarget(), { enableDiagnostics: false });
    assert.equal(compiled.diagnostics, null);
  });
});
