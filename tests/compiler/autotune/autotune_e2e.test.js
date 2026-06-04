import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget } from '../../../src/compiler/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { Autotuner } from '../../../src/compiler/autotune/autotuner.js';
import { lowerGraphToPrimFunc } from '../../../src/compiler/passes/lowering/graph_to_tensor.js';

const f32 = ScalarType.F32;

function approxEqual(a, b, eps = 1e-3) {
  return Math.abs(a - b) < eps;
}

function randArray(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2;
  return a;
}

function refMatmul(A, B, M, K, N) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  return C;
}

describe('Autotune through full pipeline: elementwise', () => {
  it('vector add with random search autotune produces correct results', () => {
    const N = 64;
    const func = buildFunction('vec_add',
      [new TensorType([N], f32), new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: { strategy: 'random', numTrials: 4, seed: 123 },
      enableFusion: false,
      enableEpilogueFusion: false,
    });
    const X = RuntimeTensor.fromArray(randArray(N), [N]);
    const Y = RuntimeTensor.fromArray(randArray(N), [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('vec_add', X, Y, out);
    for (let i = 0; i < N; i++) {
      assert.ok(approxEqual(out.data[i], X.data[i] + Y.data[i]),
        `[${i}]: ${out.data[i]} != ${X.data[i] + Y.data[i]}`);
    }
  });

  it('elementwise chain with evolutionary search autotune', () => {
    const N = 128;
    const func = buildFunction('chain',
      [new TensorType([N], f32), new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x, y]) => {
        const a = b.add(x, y);
        const m = b.mul(a.getResult(0), x);
        b.returnOp([b.exp(m.getResult(0)).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: { strategy: 'evolutionary', numTrials: 8, populationSize: 4, numGenerations: 2, seed: 42 },
      enableFusion: false,
      enableEpilogueFusion: false,
    });
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: N }, (_, i) => i * 0.01), [N]);
    const Y = RuntimeTensor.fromArray(Float32Array.from({ length: N }, () => 0.01), [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('chain', X, Y, out);
    for (let i = 0; i < N; i++) {
      const expected = Math.exp((X.data[i] + Y.data[i]) * X.data[i]);
      assert.ok(approxEqual(out.data[i], expected, 0.01),
        `[${i}]: ${out.data[i]} != ${expected}`);
    }
  });
});

describe('Autotune through full pipeline: matmul', () => {
  it('matmul with autotune produces correct results', () => {
    const M = 32, K = 16, N = 32;
    const func = buildFunction('mm',
      [new TensorType([M, K], f32), new TensorType([K, N], f32)],
      [new TensorType([M, N], f32)],
      (b, [x, w]) => {
        b.returnOp([b.matmul(x, w).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: { strategy: 'random', numTrials: 4, seed: 77 },
      enableFusion: false,
      enableEpilogueFusion: false,
    });
    const Xd = randArray(M * K);
    const Wd = randArray(K * N);
    const out = RuntimeTensor.zeros([M, N]);
    compiled.run('mm', RuntimeTensor.fromArray(Xd, [M, K]), RuntimeTensor.fromArray(Wd, [K, N]), out);
    const ref = refMatmul(Xd, Wd, M, K, N);
    for (let i = 0; i < M * N; i++) {
      assert.ok(approxEqual(out.data[i], ref[i]),
        `mm[${i}]: ${out.data[i]} != ${ref[i]}`);
    }
  });
});

describe('Autotune through full pipeline: reduction', () => {
  it('reduce sum with autotune produces correct results', () => {
    const rows = 8, cols = 16;
    const func = buildFunction('rowsum',
      [new TensorType([rows, cols], f32)],
      [new TensorType([rows], f32)],
      (b, [x]) => {
        const init = b.scalarConstant(0, f32);
        b.returnOp([b.reduce(x, init.getResult(0), [1], 'sum').getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: { strategy: 'random', numTrials: 4, seed: 99 },
      enableFusion: false,
      enableEpilogueFusion: false,
    });
    const data = randArray(rows * cols);
    const X = RuntimeTensor.fromArray(data, [rows, cols]);
    const out = RuntimeTensor.zeros([rows]);
    compiled.run('rowsum', X, out);
    for (let r = 0; r < rows; r++) {
      let expected = 0;
      for (let c = 0; c < cols; c++) expected += data[r * cols + c];
      assert.ok(approxEqual(out.data[r], expected),
        `row ${r}: ${out.data[r]} != ${expected}`);
    }
  });
});

describe('Autotune with benchmark: elementwise', () => {
  it('benchmark-refined autotune still produces correct results', () => {
    const N = 256;
    const func = buildFunction('bench_add',
      [new TensorType([N], f32), new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x, y]) => {
        b.returnOp([b.add(x, y).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: {
        strategy: 'random',
        numTrials: 4,
        seed: 55,
        enableBenchmark: true,
        benchmarkWarmup: 1,
        benchmarkRepeat: 3,
        topKForBenchmark: 2,
      },
      enableFusion: false,
      enableEpilogueFusion: false,
    });
    const X = RuntimeTensor.fromArray(randArray(N), [N]);
    const Y = RuntimeTensor.fromArray(randArray(N), [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('bench_add', X, Y, out);
    for (let i = 0; i < N; i++) {
      assert.ok(approxEqual(out.data[i], X.data[i] + Y.data[i]),
        `bench[${i}]: ${out.data[i]} != ${X.data[i] + Y.data[i]}`);
    }
  });
});

describe('Autotune DB persistence across compilations', () => {
  it('second compile with same autotuner hits cache', () => {
    const target = CPUTarget();
    const tuner = new Autotuner(target, { strategy: 'random', numTrials: 4, seed: 42 });

    const build = () => buildFunction('cached',
      [new TensorType([64], f32)],
      [new TensorType([64], f32)],
      (b, [x]) => { b.returnOp([b.exp(x).getResult(0)]); }
    );

    const pf1 = lowerGraphToPrimFunc(build());
    const result1 = tuner.tune(pf1);
    assert.ok(result1.size > 0);
    for (const [, r] of result1) assert.ok(!r.fromCache);

    const pf2 = lowerGraphToPrimFunc(build());
    const result2 = tuner.tune(pf2);
    assert.ok(result2.size > 0);
    for (const [, r] of result2) assert.ok(r.fromCache, 'second tune should use cached result');
  });
});

describe('Autotune correctness: autotuned vs untuned produce same output', () => {
  it('MLP layer: autotuned matches baseline', () => {
    const B = 4, D = 8, H = 16;
    const build = () => buildFunction('mlp_layer',
      [new TensorType([B, D], f32), new TensorType([D, H], f32), new TensorType([H], f32)],
      [new TensorType([B, H], f32)],
      (b, [x, w, bias]) => {
        const h = b.matmul(x, w);
        const bc = b.broadcast(bias, [B, H], [1]);
        const biased = b.add(h.getResult(0), bc.getResult(0));
        b.returnOp([b.relu(biased.getResult(0)).getResult(0)]);
      }
    );

    const baseline = compileGraph(build(), CPUTarget(), {
      enableAutotune: false,
      enableSchedule: false,
      enableFusion: false,
      enableEpilogueFusion: false,
    });
    const autotuned = compileGraph(build(), CPUTarget(), {
      enableAutotune: true,
      autotuneConfig: { strategy: 'random', numTrials: 4, seed: 42 },
      enableFusion: false,
      enableEpilogueFusion: false,
    });

    const X = RuntimeTensor.fromArray(randArray(B * D), [B, D]);
    const W = RuntimeTensor.fromArray(randArray(D * H), [D, H]);
    const Bi = RuntimeTensor.fromArray(randArray(H), [H]);
    const out1 = RuntimeTensor.zeros([B, H]);
    const out2 = RuntimeTensor.zeros([B, H]);
    baseline.run('mlp_layer', X, W, Bi, out1);
    autotuned.run('mlp_layer', X, W, Bi, out2);
    for (let i = 0; i < B * H; i++) {
      assert.ok(approxEqual(out1.data[i], out2.data[i]),
        `mlp[${i}]: baseline=${out1.data[i]} autotuned=${out2.data[i]}`);
    }
  });
});
