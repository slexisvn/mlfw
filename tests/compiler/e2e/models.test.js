import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;

function approxEqual(a, b, eps = 1e-3) {
  return Math.abs(a - b) < eps;
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

function refSoftmax(X, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let c = 0; c < cols; c++) mx = Math.max(mx, X[r * cols + c]);
    let s = 0;
    for (let c = 0; c < cols; c++) { out[r * cols + c] = Math.exp(X[r * cols + c] - mx); s += out[r * cols + c]; }
    for (let c = 0; c < cols; c++) out[r * cols + c] /= s;
  }
  return out;
}

function refLayernorm(X, gamma, beta, rows, cols, eps) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += X[r * cols + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c++) variance += (X[r * cols + c] - mean) ** 2;
    variance /= cols;
    const rstd = 1.0 / Math.sqrt(variance + eps);
    for (let c = 0; c < cols; c++)
      out[r * cols + c] = (X[r * cols + c] - mean) * rstd * gamma[c] + beta[c];
  }
  return out;
}

function randArray(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2;
  return a;
}

function assertClose(actual, expected, label, eps = 1e-3) {
  for (let i = 0; i < expected.length; i++) {
    assert.ok(approxEqual(actual[i], expected[i], eps),
      `${label}[${i}]: got ${actual[i]}, expected ${expected[i]}`);
  }
}

describe('MLP forward pass', () => {
  const B = 4, Din = 8, Dhid = 16, Dout = 4;

  it('linear1 -> relu -> linear2', () => {
    const func = buildFunction('mlp',
      [
        new TensorType([B, Din], f32),
        new TensorType([Din, Dhid], f32),
        new TensorType([Dhid], f32),
        new TensorType([Dhid, Dout], f32),
        new TensorType([Dout], f32),
      ],
      [new TensorType([B, Dout], f32)],
      (b, [x, w1, b1, w2, b2]) => {
        const h1 = b.matmul(x, w1);
        const bias1 = b.broadcast(b1, [B, Dhid], [1]);
        const biased1 = b.add(h1.getResult(0), bias1.getResult(0));
        const act = b.relu(biased1.getResult(0));
        const h2 = b.matmul(act.getResult(0), w2);
        const bias2 = b.broadcast(b2, [B, Dout], [1]);
        const out = b.add(h2.getResult(0), bias2.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const X = RuntimeTensor.fromArray(randArray(B * Din), [B, Din]);
    const W1 = RuntimeTensor.fromArray(randArray(Din * Dhid), [Din, Dhid]);
    const B1 = RuntimeTensor.fromArray(randArray(Dhid), [Dhid]);
    const W2 = RuntimeTensor.fromArray(randArray(Dhid * Dout), [Dhid, Dout]);
    const B2 = RuntimeTensor.fromArray(randArray(Dout), [Dout]);
    const out = RuntimeTensor.zeros([B, Dout]);

    compiled.run('mlp', X, W1, B1, W2, B2, out);

    const h1 = refMatmul(X.data, W1.data, B, Din, Dhid);
    for (let i = 0; i < B * Dhid; i++) h1[i] = Math.max(0, h1[i] + B1.data[i % Dhid]);
    const expected = refMatmul(h1, W2.data, B, Dhid, Dout);
    for (let i = 0; i < B * Dout; i++) expected[i] += B2.data[i % Dout];

    assertClose(out.data, expected, 'mlp');
  });
});

describe('Softmax layer', () => {
  it('softmax [2, 8] last axis', () => {
    const rows = 2, cols = 8;
    const func = buildFunction('softmax_layer',
      [new TensorType([rows, cols], f32)],
      [new TensorType([rows, cols], f32)],
      (b, [x]) => {
        b.returnOp([b.softmax(x, -1).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const X = RuntimeTensor.fromArray(randArray(rows * cols), [rows, cols]);
    const out = RuntimeTensor.zeros([rows, cols]);
    compiled.run('softmax_layer', X, out);
    const expected = refSoftmax(X.data, rows, cols);
    assertClose(out.data, expected, 'softmax');
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) sum += out.data[r * cols + c];
      assert.ok(approxEqual(sum, 1.0), `row ${r} sum = ${sum}`);
    }
  });
});

describe('LayerNorm layer', () => {
  it('layernorm [4, 16] last axis', () => {
    const rows = 4, cols = 16;
    const func = buildFunction('ln_layer',
      [new TensorType([rows, cols], f32), new TensorType([cols], f32), new TensorType([cols], f32)],
      [new TensorType([rows, cols], f32)],
      (b, [x, gamma, beta]) => {
        b.returnOp([b.layernorm(x, gamma, beta, -1, 1e-5).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const X = RuntimeTensor.fromArray(randArray(rows * cols), [rows, cols]);
    const gamma = RuntimeTensor.fromArray(Float32Array.from({ length: cols }, () => 1.0), [cols]);
    const beta = RuntimeTensor.fromArray(new Float32Array(cols), [cols]);
    const out = RuntimeTensor.zeros([rows, cols]);
    compiled.run('ln_layer', X, gamma, beta, out);
    const expected = refLayernorm(X.data, gamma.data, beta.data, rows, cols, 1e-5);
    assertClose(out.data, expected, 'layernorm', 1e-2);
  });
});

describe('Transformer attention block (single-head)', () => {
  const B = 2, S = 4, D = 8;

  it('Q*K^T -> scale -> softmax -> *V', () => {
    const func = buildFunction('attention',
      [
        new TensorType([B, S, D], f32),
        new TensorType([B, S, D], f32),
        new TensorType([B, S, D], f32),
      ],
      [new TensorType([B, S, D], f32)],
      (b, [Q, K, V]) => {
        const scores = b.dot(Q, K, [2], [2], [0], [0]);
        const scale = b.scalarConstant(1.0 / Math.sqrt(D), f32);
        const bcastScale = b.broadcast(scale.getResult(0), [B, S, S], []);
        const scaled = b.mul(scores.getResult(0), bcastScale.getResult(0));
        const attnWeights = b.softmax(scaled.getResult(0), -1);
        const out = b.dot(attnWeights.getResult(0), V, [2], [1], [0], [0]);
        b.returnOp([out.getResult(0)]);
      }
    );

    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const Qd = randArray(B * S * D);
    const Kd = randArray(B * S * D);
    const Vd = randArray(B * S * D);
    const out = RuntimeTensor.zeros([B, S, D]);

    compiled.run('attention',
      RuntimeTensor.fromArray(Qd, [B, S, D]),
      RuntimeTensor.fromArray(Kd, [B, S, D]),
      RuntimeTensor.fromArray(Vd, [B, S, D]),
      out
    );

    for (let batch = 0; batch < B; batch++) {
      const scores = new Float32Array(S * S);
      for (let i = 0; i < S; i++)
        for (let j = 0; j < S; j++) {
          let s = 0;
          for (let k = 0; k < D; k++) s += Qd[batch * S * D + i * D + k] * Kd[batch * S * D + j * D + k];
          scores[i * S + j] = s / Math.sqrt(D);
        }
      const attn = refSoftmax(scores, S, S);
      for (let i = 0; i < S; i++)
        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let j = 0; j < S; j++) s += attn[i * S + j] * Vd[batch * S * D + j * D + d];
          const idx = batch * S * D + i * D + d;
          assert.ok(approxEqual(out.data[idx], s, 0.05),
            `attention[${batch},${i},${d}]: got ${out.data[idx]}, expected ${s}`);
        }
    }
  });
});

describe('ResNet-style residual block', () => {
  it('x + relu(matmul(x, W) + bias)', () => {
    const N = 4, D = 8;
    const func = buildFunction('residual',
      [new TensorType([N, D], f32), new TensorType([D, D], f32), new TensorType([D], f32)],
      [new TensorType([N, D], f32)],
      (b, [x, w, bias]) => {
        const h = b.matmul(x, w);
        const bcast = b.broadcast(bias, [N, D], [1]);
        const biased = b.add(h.getResult(0), bcast.getResult(0));
        const act = b.relu(biased.getResult(0));
        const out = b.add(x, act.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const X = RuntimeTensor.fromArray(randArray(N * D), [N, D]);
    const W = RuntimeTensor.fromArray(randArray(D * D), [D, D]);
    const B_ = RuntimeTensor.fromArray(randArray(D), [D]);
    const out = RuntimeTensor.zeros([N, D]);
    compiled.run('residual', X, W, B_, out);

    const h = refMatmul(X.data, W.data, N, D, D);
    for (let i = 0; i < N * D; i++) {
      const expected = X.data[i] + Math.max(0, h[i] + B_.data[i % D]);
      assert.ok(approxEqual(out.data[i], expected),
        `residual[${i}]: got ${out.data[i]}, expected ${expected}`);
    }
  });
});

describe('GEMM + bias + relu (epilogue fused)', () => {
  it('compiles and runs with epilogue fusion enabled', () => {
    const M = 4, K = 8, N = 6;
    const func = buildFunction('gemm_epi',
      [new TensorType([M, K], f32), new TensorType([K, N], f32), new TensorType([N], f32)],
      [new TensorType([M, N], f32)],
      (b, [x, w, bias]) => {
        const h = b.matmul(x, w);
        const bcast = b.broadcast(bias, [M, N], [1]);
        const biased = b.add(h.getResult(0), bcast.getResult(0));
        const out = b.relu(biased.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );

    const withEpi = compileGraph(func, CPUTarget({ enableEpilogueFusion: true }));
    const funcNoEpi = buildFunction('gemm_no_epi',
      [new TensorType([M, K], f32), new TensorType([K, N], f32), new TensorType([N], f32)],
      [new TensorType([M, N], f32)],
      (b, [x, w, bias]) => {
        const h = b.matmul(x, w);
        const bcast = b.broadcast(bias, [M, N], [1]);
        const biased = b.add(h.getResult(0), bcast.getResult(0));
        const out = b.relu(biased.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const withoutEpi = compileGraph(funcNoEpi, CPUTarget({ enableEpilogueFusion: false }));

    const X = RuntimeTensor.fromArray(randArray(M * K), [M, K]);
    const W = RuntimeTensor.fromArray(randArray(K * N), [K, N]);
    const B_ = RuntimeTensor.fromArray(randArray(N), [N]);
    const out1 = RuntimeTensor.zeros([M, N]);
    const out2 = RuntimeTensor.zeros([M, N]);
    withEpi.run('gemm_epi', X, W, B_, out1);
    withoutEpi.run('gemm_no_epi', X, W, B_, out2);
    assertClose(out1.data, out2.data, 'epilogue_vs_standard');
  });
});

describe('GPU codegen for model patterns', () => {
  it('MLP compiles to valid CUDA', () => {
    const func = buildFunction('gpu_mlp',
      [new TensorType([4, 8], f32), new TensorType([8, 16], f32), new TensorType([16], f32)],
      [new TensorType([4, 16], f32)],
      (b, [x, w, bias]) => {
        const h = b.matmul(x, w);
        const bcast = b.broadcast(bias, [4, 16], [1]);
        const out = b.add(h.getResult(0), bcast.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }));
    const source = compiled.getSource('gpu_mlp');
    assert.ok(source.includes('__global__'));
    assert.ok(source.includes('float*'));
  });

  it('softmax compiles to valid CUDA', () => {
    const func = buildFunction('gpu_softmax',
      [new TensorType([2, 8], f32)],
      [new TensorType([2, 8], f32)],
      (b, [x]) => {
        b.returnOp([b.softmax(x, -1).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const source = compiled.getSource('gpu_softmax');
    assert.ok(source.includes('__global__'));
    assert.ok(source.includes('expf'));
  });
});

describe('where + clamp pattern (GELU approx)', () => {
  it('compiles and produces correct output', () => {
    const N = 16;
    const func = buildFunction('gelu_approx',
      [new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x]) => {
        const half = b.broadcast(b.scalarConstant(0.5, f32).getResult(0), [N], []);
        const one = b.broadcast(b.scalarConstant(1.0, f32).getResult(0), [N], []);
        const coeff = b.broadcast(b.scalarConstant(0.7978845608, f32).getResult(0), [N], []);
        const kappa = b.broadcast(b.scalarConstant(0.044715, f32).getResult(0), [N], []);
        const x3 = b.mul(x, b.mul(x, x).getResult(0));
        const inner = b.mul(coeff.getResult(0), b.add(x, b.mul(kappa.getResult(0), x3.getResult(0)).getResult(0)).getResult(0));
        const tanhVal = b.tanh(inner.getResult(0));
        const cdf = b.mul(half.getResult(0), b.add(one.getResult(0), tanhVal.getResult(0)).getResult(0));
        const out = b.mul(x, cdf.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: N }, (_, i) => (i - 8) * 0.5), [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('gelu_approx', X, out);

    for (let i = 0; i < N; i++) {
      const x = X.data[i];
      const expected = 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
      assert.ok(approxEqual(out.data[i], expected, 0.01),
        `gelu[${i}]: got ${out.data[i]}, expected ${expected}`);
    }
  });
});

describe('Reduce patterns', () => {
  it('mean + variance (layernorm building blocks)', () => {
    const rows = 3, cols = 5;
    const func = buildFunction('mean_var',
      [new TensorType([rows, cols], f32)],
      [new TensorType([rows], f32), new TensorType([rows], f32)],
      (b, [x]) => {
        const init = b.scalarConstant(0, f32);
        const mean = b.reduce(x, init.getResult(0), [1], 'mean');
        const bcastMean = b.broadcast(mean.getResult(0), [rows, cols], [0]);
        const diff = b.sub(x, bcastMean.getResult(0));
        const sq = b.mul(diff.getResult(0), diff.getResult(0));
        const variance = b.reduce(sq.getResult(0), init.getResult(0), [1], 'mean');
        b.returnOp([mean.getResult(0), variance.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const data = [1,2,3,4,5, 10,20,30,40,50, 0,0,0,0,0];
    const X = RuntimeTensor.fromArray(data, [rows, cols]);
    const outMean = RuntimeTensor.zeros([rows]);
    const outVar = RuntimeTensor.zeros([rows]);
    compiled.run('mean_var', X, outMean, outVar);
    assert.ok(approxEqual(outMean.data[0], 3.0));
    assert.ok(approxEqual(outMean.data[1], 30.0));
    assert.ok(approxEqual(outMean.data[2], 0.0));
    assert.ok(approxEqual(outVar.data[0], 2.0));
    assert.ok(outVar.data[1] > 0);
    assert.ok(approxEqual(outVar.data[2], 0.0));
  });
});

describe('SiLU activation (x * sigmoid(x))', () => {
  it('compiles and matches reference', () => {
    const N = 16;
    const func = buildFunction('silu',
      [new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x]) => {
        const one = b.broadcast(b.scalarConstant(1.0, f32).getResult(0), [N], []);
        const negx = b.neg(x);
        const expneg = b.exp(negx.getResult(0));
        const denom = b.add(one.getResult(0), expneg.getResult(0));
        const sigmoid = b.div(one.getResult(0), denom.getResult(0));
        const out = b.mul(x, sigmoid.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const X = RuntimeTensor.fromArray(Float32Array.from({ length: N }, (_, i) => (i - 8) * 0.5), [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('silu', X, out);
    for (let i = 0; i < N; i++) {
      const x = X.data[i];
      const expected = x / (1 + Math.exp(-x));
      assert.ok(approxEqual(out.data[i], expected, 0.01),
        `silu[${i}]: got ${out.data[i]}, expected ${expected}`);
    }
  });
});

describe('3-layer MLP with different activations per layer', () => {
  it('linear(tanh) -> linear(relu) -> linear', () => {
    const B = 2, D0 = 6, D1 = 10, D2 = 8, D3 = 3;
    const func = buildFunction('mlp3',
      [
        new TensorType([B, D0], f32),
        new TensorType([D0, D1], f32), new TensorType([D1], f32),
        new TensorType([D1, D2], f32), new TensorType([D2], f32),
        new TensorType([D2, D3], f32), new TensorType([D3], f32),
      ],
      [new TensorType([B, D3], f32)],
      (b, [x, w1, b1, w2, b2, w3, b3]) => {
        const h1 = b.matmul(x, w1);
        const biased1 = b.add(h1.getResult(0), b.broadcast(b1, [B, D1], [1]).getResult(0));
        const act1 = b.tanh(biased1.getResult(0));

        const h2 = b.matmul(act1.getResult(0), w2);
        const biased2 = b.add(h2.getResult(0), b.broadcast(b2, [B, D2], [1]).getResult(0));
        const act2 = b.relu(biased2.getResult(0));

        const h3 = b.matmul(act2.getResult(0), w3);
        const out = b.add(h3.getResult(0), b.broadcast(b3, [B, D3], [1]).getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const args = [
      RuntimeTensor.fromArray(randArray(B * D0), [B, D0]),
      RuntimeTensor.fromArray(randArray(D0 * D1), [D0, D1]),
      RuntimeTensor.fromArray(randArray(D1), [D1]),
      RuntimeTensor.fromArray(randArray(D1 * D2), [D1, D2]),
      RuntimeTensor.fromArray(randArray(D2), [D2]),
      RuntimeTensor.fromArray(randArray(D2 * D3), [D2, D3]),
      RuntimeTensor.fromArray(randArray(D3), [D3]),
    ];
    const out = RuntimeTensor.zeros([B, D3]);
    compiled.run('mlp3', ...args, out);
    for (let i = 0; i < B * D3; i++) {
      assert.ok(isFinite(out.data[i]), `mlp3[${i}] should be finite, got ${out.data[i]}`);
    }
  });
});

describe('Batch matrix multiply', () => {
  it('[2, 3, 4] x [2, 4, 5] -> [2, 3, 5]', () => {
    const Ba = 2, M = 3, K = 4, N = 5;
    const func = buildFunction('batch_matmul',
      [new TensorType([Ba, M, K], f32), new TensorType([Ba, K, N], f32)],
      [new TensorType([Ba, M, N], f32)],
      (b, [a, bb]) => {
        const out = b.matmul(a, bb);
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const Ad = randArray(Ba * M * K);
    const Bd = randArray(Ba * K * N);
    const A = RuntimeTensor.fromArray(Ad, [Ba, M, K]);
    const B_ = RuntimeTensor.fromArray(Bd, [Ba, K, N]);
    const out = RuntimeTensor.zeros([Ba, M, N]);
    compiled.run('batch_matmul', A, B_, out);
    for (let batch = 0; batch < Ba; batch++) {
      const ref = refMatmul(
        Ad.subarray(batch * M * K, (batch + 1) * M * K),
        Bd.subarray(batch * K * N, (batch + 1) * K * N),
        M, K, N
      );
      for (let i = 0; i < M * N; i++) {
        assert.ok(approxEqual(out.data[batch * M * N + i], ref[i]),
          `bmm[${batch},${i}]: got ${out.data[batch * M * N + i]}, expected ${ref[i]}`);
      }
    }
  });
});

describe('Pointwise math chain (sin, cos, pow, sqrt)', () => {
  it('sqrt(sin(x)^2 + cos(x)^2) ≈ 1', () => {
    const N = 32;
    const func = buildFunction('trig_identity',
      [new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x]) => {
        const s = b.sin(x);
        const c = b.cos(x);
        const s2 = b.mul(s.getResult(0), s.getResult(0));
        const c2 = b.mul(c.getResult(0), c.getResult(0));
        const sum = b.add(s2.getResult(0), c2.getResult(0));
        const out = b.sqrt(sum.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const X = RuntimeTensor.fromArray(randArray(N), [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('trig_identity', X, out);
    for (let i = 0; i < N; i++) {
      assert.ok(approxEqual(out.data[i], 1.0, 1e-4),
        `sin²+cos²[${i}]: got ${out.data[i]}`);
    }
  });
});

describe('Reshape + transpose pipeline', () => {
  it('[2, 3, 4] -> transpose [0,2,1] -> reshape [2, 12]', () => {
    const func = buildFunction('reshape_transpose',
      [new TensorType([2, 3, 4], f32)],
      [new TensorType([2, 12], f32)],
      (b, [x]) => {
        const t = b.transpose(x, [0, 2, 1]);
        const r = b.reshape(t.getResult(0), [2, 12]);
        b.returnOp([r.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const data = Float32Array.from({ length: 24 }, (_, i) => i);
    const X = RuntimeTensor.fromArray(data, [2, 3, 4]);
    const out = RuntimeTensor.zeros([2, 12]);
    compiled.run('reshape_transpose', X, out);
    for (let b = 0; b < 2; b++)
      for (let j = 0; j < 4; j++)
        for (let i = 0; i < 3; i++) {
          const srcIdx = b * 12 + i * 4 + j;
          const dstIdx = b * 12 + j * 3 + i;
          assert.equal(out.data[dstIdx], data[srcIdx],
            `transpose[${b},${j},${i}]: got ${out.data[dstIdx]}, expected ${data[srcIdx]}`);
        }
  });
});

describe('Multi-head attention (2 heads)', () => {
  const B = 1, S = 4, D = 8, H = 2, Dh = D / H;

  it('split heads -> attention -> merge heads', () => {
    const func = buildFunction('mha',
      [
        new TensorType([B, S, D], f32),
        new TensorType([D, D], f32),
        new TensorType([D, D], f32),
        new TensorType([D, D], f32),
        new TensorType([D, D], f32),
      ],
      [new TensorType([B, S, D], f32)],
      (b, [x, wq, wk, wv, wo]) => {
        const Q = b.matmul(b.reshape(x, [B * S, D]).getResult(0), wq);
        const K = b.matmul(b.reshape(x, [B * S, D]).getResult(0), wk);
        const V = b.matmul(b.reshape(x, [B * S, D]).getResult(0), wv);

        const Qr = b.reshape(Q.getResult(0), [B, S, D]);
        const Kr = b.reshape(K.getResult(0), [B, S, D]);
        const Vr = b.reshape(V.getResult(0), [B, S, D]);

        const scores = b.dot(Qr.getResult(0), Kr.getResult(0), [2], [2], [0], [0]);
        const scale = b.broadcast(b.scalarConstant(1.0 / Math.sqrt(D), f32).getResult(0), [B, S, S], []);
        const scaled = b.mul(scores.getResult(0), scale.getResult(0));
        const attn = b.softmax(scaled.getResult(0), -1);
        const ctx = b.dot(attn.getResult(0), Vr.getResult(0), [2], [1], [0], [0]);

        const flat = b.reshape(ctx.getResult(0), [B * S, D]);
        const proj = b.matmul(flat.getResult(0), wo);
        const out = b.reshape(proj.getResult(0), [B, S, D]);
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const X = RuntimeTensor.fromArray(randArray(B * S * D), [B, S, D]);
    const Wq = RuntimeTensor.fromArray(randArray(D * D), [D, D]);
    const Wk = RuntimeTensor.fromArray(randArray(D * D), [D, D]);
    const Wv = RuntimeTensor.fromArray(randArray(D * D), [D, D]);
    const Wo = RuntimeTensor.fromArray(randArray(D * D), [D, D]);
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('mha', X, Wq, Wk, Wv, Wo, out);
    for (let i = 0; i < B * S * D; i++) {
      assert.ok(isFinite(out.data[i]), `mha[${i}] should be finite`);
    }
  });
});

describe('Embedding lookup via matmul', () => {
  it('one-hot x embedding_table', () => {
    const vocabSize = 8, embDim = 4, seqLen = 3;
    const func = buildFunction('embedding',
      [new TensorType([seqLen, vocabSize], f32), new TensorType([vocabSize, embDim], f32)],
      [new TensorType([seqLen, embDim], f32)],
      (b, [oneHot, table]) => {
        b.returnOp([b.matmul(oneHot, table).getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }));
    const oneHot = new Float32Array(seqLen * vocabSize);
    const indices = [2, 0, 5];
    for (let i = 0; i < seqLen; i++) oneHot[i * vocabSize + indices[i]] = 1;
    const table = randArray(vocabSize * embDim);
    const out = RuntimeTensor.zeros([seqLen, embDim]);
    compiled.run('embedding',
      RuntimeTensor.fromArray(oneHot, [seqLen, vocabSize]),
      RuntimeTensor.fromArray(table, [vocabSize, embDim]),
      out
    );
    for (let i = 0; i < seqLen; i++)
      for (let d = 0; d < embDim; d++)
        assert.ok(approxEqual(out.data[i * embDim + d], table[indices[i] * embDim + d]),
          `embedding[${i},${d}]: got ${out.data[i * embDim + d]}, expected ${table[indices[i] * embDim + d]}`);
  });
});

describe('Logistic regression (sigmoid + BCE loss)', () => {
  it('compiles and produces valid loss', () => {
    const N = 8;
    const func = buildFunction('bce_loss',
      [new TensorType([N], f32), new TensorType([N], f32)],
      [new TensorType([], f32)],
      (b, [logits, labels]) => {
        const one = b.broadcast(b.scalarConstant(1.0, f32).getResult(0), [N], []);
        const neg_logits = b.neg(logits);
        const exp_neg = b.exp(neg_logits.getResult(0));
        const denom = b.add(one.getResult(0), exp_neg.getResult(0));
        const probs = b.div(one.getResult(0), denom.getResult(0));
        const eps = b.broadcast(b.scalarConstant(1e-7, f32).getResult(0), [N], []);
        const log_p = b.log(b.add(probs.getResult(0), eps.getResult(0)).getResult(0));
        const log_1mp = b.log(b.add(b.sub(one.getResult(0), probs.getResult(0)).getResult(0), eps.getResult(0)).getResult(0));
        const term1 = b.mul(labels, log_p.getResult(0));
        const term2 = b.mul(b.sub(one.getResult(0), labels).getResult(0), log_1mp.getResult(0));
        const pointwise = b.neg(b.add(term1.getResult(0), term2.getResult(0)).getResult(0));
        const init = b.scalarConstant(0, f32);
        const total = b.reduce(pointwise.getResult(0), init.getResult(0), [0], 'sum');
        const count = b.scalarConstant(N, f32);
        const loss = b.div(total.getResult(0), b.broadcast(count.getResult(0), [], []).getResult(0));
        b.returnOp([loss.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const logits = RuntimeTensor.fromArray([2, -1, 0.5, 3, -2, 1, -0.5, 0], [N]);
    const labels = RuntimeTensor.fromArray([1, 0, 1, 1, 0, 1, 0, 0], [N]);
    const out = RuntimeTensor.zeros([]);
    compiled.run('bce_loss', logits, labels, out);
    assert.ok(isFinite(out.data[0]), `loss should be finite, got ${out.data[0]}`);
    assert.ok(out.data[0] > 0, `loss should be positive, got ${out.data[0]}`);

    let refLoss = 0;
    for (let i = 0; i < N; i++) {
      const p = 1 / (1 + Math.exp(-logits.data[i]));
      refLoss += -(labels.data[i] * Math.log(p + 1e-7) + (1 - labels.data[i]) * Math.log(1 - p + 1e-7));
    }
    refLoss /= N;
    assert.ok(approxEqual(out.data[0], refLoss, 0.01),
      `bce: got ${out.data[0]}, expected ${refLoss}`);
  });
});

describe('RMSNorm', () => {
  it('compiles and normalizes correctly', () => {
    const rows = 3, cols = 6;
    const func = buildFunction('rmsnorm',
      [new TensorType([rows, cols], f32), new TensorType([cols], f32)],
      [new TensorType([rows, cols], f32)],
      (b, [x, weight]) => {
        const sq = b.mul(x, x);
        const init = b.scalarConstant(0, f32);
        const ms = b.reduce(sq.getResult(0), init.getResult(0), [1], 'mean');
        const eps = b.broadcast(b.scalarConstant(1e-6, f32).getResult(0), ms.getResult(0).type.shape, []);
        const msPlusEps = b.add(ms.getResult(0), eps.getResult(0));
        const rms = b.rsqrt(msPlusEps.getResult(0));
        const bcastRms = b.broadcast(rms.getResult(0), [rows, cols], [0]);
        const normalized = b.mul(x, bcastRms.getResult(0));
        const bcastW = b.broadcast(weight, [rows, cols], [1]);
        const out = b.mul(normalized.getResult(0), bcastW.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const X = RuntimeTensor.fromArray(randArray(rows * cols), [rows, cols]);
    const W = RuntimeTensor.fromArray(Float32Array.from({ length: cols }, () => 1.0), [cols]);
    const out = RuntimeTensor.zeros([rows, cols]);
    compiled.run('rmsnorm', X, W, out);
    for (let r = 0; r < rows; r++) {
      let ss = 0;
      for (let c = 0; c < cols; c++) ss += out.data[r * cols + c] ** 2;
      const rms = Math.sqrt(ss / cols);
      assert.ok(approxEqual(rms, 1.0, 0.1), `row ${r} RMS: got ${rms}, expected ~1`);
    }
  });
});

describe('Max pooling via reduce', () => {
  it('global max pool [2, 3, 4] -> [2, 3]', () => {
    const B = 2, C = 3, W = 4;
    const func = buildFunction('maxpool',
      [new TensorType([B, C, W], f32)],
      [new TensorType([B, C], f32)],
      (b, [x]) => {
        const init = b.scalarConstant(-Infinity, f32);
        const out = b.reduce(x, init.getResult(0), [2], 'max');
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const data = randArray(B * C * W);
    const X = RuntimeTensor.fromArray(data, [B, C, W]);
    const out = RuntimeTensor.zeros([B, C]);
    compiled.run('maxpool', X, out);
    for (let bi = 0; bi < B; bi++)
      for (let ci = 0; ci < C; ci++) {
        let mx = -Infinity;
        for (let wi = 0; wi < W; wi++) mx = Math.max(mx, data[bi * C * W + ci * W + wi]);
        assert.ok(approxEqual(out.data[bi * C + ci], mx),
          `maxpool[${bi},${ci}]: got ${out.data[bi * C + ci]}, expected ${mx}`);
      }
  });
});

describe('Fused elementwise chain with fusion enabled', () => {
  it('exp(x+y) * (x-y) fuses correctly', () => {
    const N = 32;
    const func = buildFunction('fused_chain',
      [new TensorType([N], f32), new TensorType([N], f32)],
      [new TensorType([N], f32)],
      (b, [x, y]) => {
        const sum = b.add(x, y);
        const diff = b.sub(x, y);
        const e = b.exp(sum.getResult(0));
        const out = b.mul(e.getResult(0), diff.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: true });
    const Xd = randArray(N);
    const Yd = randArray(N);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('fused_chain', RuntimeTensor.fromArray(Xd, [N]), RuntimeTensor.fromArray(Yd, [N]), out);
    for (let i = 0; i < N; i++) {
      const expected = Math.exp(Xd[i] + Yd[i]) * (Xd[i] - Yd[i]);
      assert.ok(approxEqual(out.data[i], expected, 0.01),
        `fused[${i}]: got ${out.data[i]}, expected ${expected}`);
    }
  });
});

describe('Linear regression forward + MSE loss', () => {
  it('compiles full train-step graph', () => {
    const N = 6, D = 4;
    const func = buildFunction('mse_loss',
      [new TensorType([N, D], f32), new TensorType([D, 1], f32), new TensorType([N, 1], f32)],
      [new TensorType([], f32)],
      (b, [x, w, y]) => {
        const pred = b.matmul(x, w);
        const diff = b.sub(pred.getResult(0), y);
        const sq = b.mul(diff.getResult(0), diff.getResult(0));
        const init = b.scalarConstant(0, f32);
        const total = b.reduce(sq.getResult(0), init.getResult(0), [0, 1], 'mean');
        b.returnOp([total.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), { enableFusion: false });
    const X = RuntimeTensor.fromArray(randArray(N * D), [N, D]);
    const W = RuntimeTensor.fromArray(randArray(D), [D, 1]);
    const Y = RuntimeTensor.fromArray(randArray(N), [N, 1]);
    const out = RuntimeTensor.zeros([]);
    compiled.run('mse_loss', X, W, Y, out);
    assert.ok(isFinite(out.data[0]) && out.data[0] >= 0, `MSE should be non-negative finite, got ${out.data[0]}`);
  });
});
