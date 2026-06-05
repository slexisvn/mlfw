import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/backend/target.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const f32 = ScalarType.F32;
const i32 = ScalarType.I32;

function rand(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 2;
  return a;
}

function ones(n) { return Float32Array.from({ length: n }, () => 1); }
function zeros(n) { return new Float32Array(n); }

function close(a, b, eps = 1e-2) { return Math.abs(a - b) < eps; }

function assertFinite(data, label) {
  for (let i = 0; i < data.length; i++)
    assert.ok(isFinite(data[i]), `${label}[${i}] = ${data[i]} is not finite`);
}

function assertClose(actual, expected, label, eps = 1e-2) {
  for (let i = 0; i < expected.length; i++)
    assert.ok(close(actual[i], expected[i], eps), `${label}[${i}]: ${actual[i]} != ${expected[i]}`);
}

function T(shape, dtype = f32) { return new TensorType(shape, dtype); }

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget({ enableEpilogueFusion: false }), {
    enableFusion: false, enableEpilogueFusion: false, ...opts
  });
}

function compileFused(func) {
  return compile(func, { enableFusion: true });
}

function compileGPU(func) {
  return compileGraph(func, GPUTarget({ enableEpilogueFusion: false }), {
    enableFusion: false, enableEpilogueFusion: false
  });
}

describe('Transformer encoder block', () => {
  const B = 1, S = 6, D = 16;

  it('self-attention + layernorm + FFN + residual', () => {
    const func = buildFunction('encoder_block', [
      T([B, S, D]), T([D, D]), T([D, D]), T([D, D]), T([D, D]),
      T([D]), T([D]), T([D, D * 4]), T([D * 4], f32), T([D * 4, D]), T([D]),
      T([D]), T([D]),
    ], [T([B, S, D])], (b, [x, wq, wk, wv, wo, ln1g, ln1b, wff1, bff1, wff2, bff2, ln2g, ln2b]) => {
      const flat = b.reshape(x, [B * S, D]);
      const Q = b.matmul(flat.getResult(0), wq);
      const K = b.matmul(flat.getResult(0), wk);
      const V = b.matmul(flat.getResult(0), wv);
      const Qr = b.reshape(Q.getResult(0), [B, S, D]);
      const Kr = b.reshape(K.getResult(0), [B, S, D]);
      const Vr = b.reshape(V.getResult(0), [B, S, D]);
      const scores = b.dot(Qr.getResult(0), Kr.getResult(0), [2], [2], [0], [0]);
      const scale = b.broadcast(b.scalarConstant(1.0 / Math.sqrt(D), f32).getResult(0), [B, S, S], []);
      const scaled = b.mul(scores.getResult(0), scale.getResult(0));
      const attn = b.softmax(scaled.getResult(0), -1);
      const ctx = b.dot(attn.getResult(0), Vr.getResult(0), [2], [1], [0], [0]);
      const ctxFlat = b.reshape(ctx.getResult(0), [B * S, D]);
      const proj = b.matmul(ctxFlat.getResult(0), wo);
      const projR = b.reshape(proj.getResult(0), [B, S, D]);
      const res1 = b.add(x, projR.getResult(0));
      const ln1 = b.layernorm(res1.getResult(0), ln1g, ln1b, -1, 1e-5);

      const ffFlat = b.reshape(ln1.getResult(0), [B * S, D]);
      const h1 = b.matmul(ffFlat.getResult(0), wff1);
      const biased1 = b.add(h1.getResult(0), b.broadcast(bff1, [B * S, D * 4], [1]).getResult(0));
      const act = b.relu(biased1.getResult(0));
      const h2 = b.matmul(act.getResult(0), wff2);
      const biased2 = b.add(h2.getResult(0), b.broadcast(bff2, [B * S, D], [1]).getResult(0));
      const ffR = b.reshape(biased2.getResult(0), [B, S, D]);
      const res2 = b.add(ln1.getResult(0), ffR.getResult(0));
      const out = b.layernorm(res2.getResult(0), ln2g, ln2b, -1, 1e-5);
      b.returnOp([out.getResult(0)]);
    });

    const compiled = compile(func);
    const args = [
      RuntimeTensor.fromArray(rand(B * S * D), [B, S, D]),
      ...Array.from({ length: 4 }, () => RuntimeTensor.fromArray(rand(D * D), [D, D])),
      RuntimeTensor.fromArray(ones(D), [D]), RuntimeTensor.fromArray(zeros(D), [D]),
      RuntimeTensor.fromArray(rand(D * D * 4), [D, D * 4]),
      RuntimeTensor.fromArray(rand(D * 4), [D * 4]),
      RuntimeTensor.fromArray(rand(D * 4 * D), [D * 4, D]),
      RuntimeTensor.fromArray(rand(D), [D]),
      RuntimeTensor.fromArray(ones(D), [D]), RuntimeTensor.fromArray(zeros(D), [D]),
    ];
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('encoder_block', ...args, out);
    assertFinite(out.data, 'encoder');
  });
});

describe('GPT-2 style decoder layer', () => {
  const B = 1, S = 4, D = 8;

  it('causal self-attention + FFN + 2x residual + 2x layernorm', () => {
    const func = buildFunction('decoder_layer', [
      T([B, S, D]),
      T([D, D]), T([D, D]), T([D, D]), T([D, D]),
      T([D]), T([D]),
      T([D, D * 4]), T([D * 4]),
      T([D * 4, D]), T([D]),
      T([D]), T([D]),
    ], [T([B, S, D])], (b, [x, wq, wk, wv, wo, ln1g, ln1b, wff1, bff1, wff2, bff2, ln2g, ln2b]) => {
      const ln1 = b.layernorm(x, ln1g, ln1b, -1, 1e-5);
      const flat = b.reshape(ln1.getResult(0), [B * S, D]);
      const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
      const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
      const V = b.reshape(b.matmul(flat.getResult(0), wv).getResult(0), [B, S, D]);
      const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
      const scale = b.broadcast(b.scalarConstant(1.0 / Math.sqrt(D), f32).getResult(0), [B, S, S], []);
      const attn = b.softmax(b.mul(scores.getResult(0), scale.getResult(0)).getResult(0), -1);
      const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
      const proj = b.reshape(b.matmul(b.reshape(ctx.getResult(0), [B * S, D]).getResult(0), wo).getResult(0), [B, S, D]);
      const res1 = b.add(x, proj.getResult(0));

      const ln2 = b.layernorm(res1.getResult(0), ln2g, ln2b, -1, 1e-5);
      const ffFlat = b.reshape(ln2.getResult(0), [B * S, D]);
      const h1 = b.add(b.matmul(ffFlat.getResult(0), wff1).getResult(0), b.broadcast(bff1, [B * S, D * 4], [1]).getResult(0));
      const act = b.relu(h1.getResult(0));
      const h2 = b.add(b.matmul(act.getResult(0), wff2).getResult(0), b.broadcast(bff2, [B * S, D], [1]).getResult(0));
      const out = b.add(res1.getResult(0), b.reshape(h2.getResult(0), [B, S, D]).getResult(0));
      b.returnOp([out.getResult(0)]);
    });

    const compiled = compile(func);
    const args = [
      RuntimeTensor.fromArray(rand(B * S * D), [B, S, D]),
      ...Array.from({ length: 4 }, () => RuntimeTensor.fromArray(rand(D * D), [D, D])),
      RuntimeTensor.fromArray(ones(D), [D]), RuntimeTensor.fromArray(zeros(D), [D]),
      RuntimeTensor.fromArray(rand(D * D * 4), [D, D * 4]),
      RuntimeTensor.fromArray(rand(D * 4), [D * 4]),
      RuntimeTensor.fromArray(rand(D * 4 * D), [D * 4, D]),
      RuntimeTensor.fromArray(rand(D), [D]),
      RuntimeTensor.fromArray(ones(D), [D]), RuntimeTensor.fromArray(zeros(D), [D]),
    ];
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('decoder_layer', ...args, out);
    assertFinite(out.data, 'decoder');
  });
});

describe('U-Net style encoder path', () => {
  it('3 stages: linear+relu downsample', () => {
    const func = buildFunction('unet_enc', [
      T([2, 32]), T([32, 16]), T([16]), T([16, 8]), T([8]), T([8, 4]), T([4]),
    ], [T([2, 4])], (b, [x, w1, b1, w2, b2, w3, b3]) => {
      const h1 = b.relu(b.add(b.matmul(x, w1).getResult(0), b.broadcast(b1, [2, 16], [1]).getResult(0)).getResult(0));
      const h2 = b.relu(b.add(b.matmul(h1.getResult(0), w2).getResult(0), b.broadcast(b2, [2, 8], [1]).getResult(0)).getResult(0));
      const h3 = b.add(b.matmul(h2.getResult(0), w3).getResult(0), b.broadcast(b3, [2, 4], [1]).getResult(0));
      b.returnOp([h3.getResult(0)]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([2, 4]);
    compiled.run('unet_enc',
      RuntimeTensor.fromArray(rand(64), [2, 32]),
      RuntimeTensor.fromArray(rand(512), [32, 16]), RuntimeTensor.fromArray(rand(16), [16]),
      RuntimeTensor.fromArray(rand(128), [16, 8]), RuntimeTensor.fromArray(rand(8), [8]),
      RuntimeTensor.fromArray(rand(32), [8, 4]), RuntimeTensor.fromArray(rand(4), [4]),
      out
    );
    assertFinite(out.data, 'unet_enc');
  });
});

describe('Mixture of Experts gate (simplified)', () => {
  it('softmax gate -> weighted expert output', () => {
    const B = 2, D = 8, E = 4;
    const func = buildFunction('moe_gate', [
      T([B, D]), T([D, E]), T([D, D]),
    ], [T([B, D])], (b, [x, wGate, expert]) => {
      const logits = b.matmul(x, wGate);
      const weights = b.softmax(logits.getResult(0), -1);
      const maxW = b.reduce(weights.getResult(0), b.scalarConstant(-Infinity, f32).getResult(0), [1], 'max');
      const scale = b.broadcast(maxW.getResult(0), [B, D], [0]);
      const expertOut = b.matmul(x, expert);
      const out = b.mul(expertOut.getResult(0), scale.getResult(0));
      b.returnOp([out.getResult(0)]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([B, D]);
    compiled.run('moe_gate',
      RuntimeTensor.fromArray(rand(B * D), [B, D]),
      RuntimeTensor.fromArray(rand(D * E), [D, E]),
      RuntimeTensor.fromArray(rand(D * D), [D, D]),
      out
    );
    assertFinite(out.data, 'moe');
  });
});

describe('Contrastive loss (SimCLR-style)', () => {
  it('normalize + similarity matrix + cross-entropy', () => {
    const N = 4, D = 8;
    const func = buildFunction('contrastive', [
      T([N, D]), T([N, D]),
    ], [T([])], (b, [z1, z2]) => {
      const sq1 = b.mul(z1, z1);
      const norm1_sq = b.reduce(sq1.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      const norm1 = b.sqrt(b.add(norm1_sq.getResult(0), b.broadcast(b.scalarConstant(1e-8, f32).getResult(0), [N], []).getResult(0)).getResult(0));
      const n1 = b.div(z1, b.broadcast(norm1.getResult(0), [N, D], [0]).getResult(0));

      const sq2 = b.mul(z2, z2);
      const norm2_sq = b.reduce(sq2.getResult(0), b.scalarConstant(0, f32).getResult(0), [1], 'sum');
      const norm2 = b.sqrt(b.add(norm2_sq.getResult(0), b.broadcast(b.scalarConstant(1e-8, f32).getResult(0), [N], []).getResult(0)).getResult(0));
      const n2 = b.div(z2, b.broadcast(norm2.getResult(0), [N, D], [0]).getResult(0));

      const sim = b.dot(n1.getResult(0), n2.getResult(0), [1], [1]);
      const temp = b.broadcast(b.scalarConstant(0.5, f32).getResult(0), [N, N], []);
      const logits = b.div(sim.getResult(0), temp.getResult(0));
      const loss = b.reduce(logits.getResult(0), b.scalarConstant(0, f32).getResult(0), [0, 1], 'mean');
      b.returnOp([loss.getResult(0)]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([]);
    compiled.run('contrastive',
      RuntimeTensor.fromArray(rand(N * D), [N, D]),
      RuntimeTensor.fromArray(rand(N * D), [N, D]),
      out
    );
    assertFinite(out.data, 'contrastive');
  });
});

describe('Long elementwise chain stress', () => {
  it('20-op chain compiles and fuses correctly', () => {
    const N = 64;
    const func = buildFunction('long_chain', [T([N]), T([N])], [T([N])], (b, [x, y]) => {
      let cur = b.add(x, y).getResult(0);
      cur = b.mul(cur, x).getResult(0);
      cur = b.sub(cur, y).getResult(0);
      cur = b.exp(b.neg(b.abs(cur).getResult(0)).getResult(0)).getResult(0);
      cur = b.add(cur, b.broadcast(b.scalarConstant(1, f32).getResult(0), [N], []).getResult(0)).getResult(0);
      cur = b.div(b.broadcast(b.scalarConstant(1, f32).getResult(0), [N], []).getResult(0), cur).getResult(0);
      cur = b.mul(cur, b.sin(x).getResult(0)).getResult(0);
      cur = b.add(cur, b.cos(y).getResult(0)).getResult(0);
      cur = b.sqrt(b.abs(cur).getResult(0)).getResult(0);
      cur = b.tanh(cur).getResult(0);
      b.returnOp([cur]);
    });
    const fused = compileFused(func);
    const plain = compile(func);
    const X = RuntimeTensor.fromArray(rand(N), [N]);
    const Y = RuntimeTensor.fromArray(rand(N), [N]);
    const out1 = RuntimeTensor.zeros([N]);
    const out2 = RuntimeTensor.zeros([N]);
    fused.run('long_chain', X, Y, out1);
    plain.run('long_chain', X, Y, out2);
    assertClose(out1.data, out2.data, 'long_chain_fused_vs_plain');
  });
});

describe('Multi-output divergent compute', () => {
  it('same input -> exp branch + neg branch', () => {
    const N = 16;
    const func = buildFunction('multi_branch', [T([N])], [T([N]), T([N])],
      (b, [x]) => {
        const pos = b.exp(x);
        const neg = b.neg(x);
        b.returnOp([pos.getResult(0), neg.getResult(0)]);
      }
    );
    const compiled = compile(func);
    const data = rand(N);
    const out1 = RuntimeTensor.zeros([N]);
    const out2 = RuntimeTensor.zeros([N]);
    compiled.run('multi_branch', RuntimeTensor.fromArray(data, [N]), out1, out2);
    for (let i = 0; i < N; i++) {
      assert.ok(close(out1.data[i], Math.exp(data[i]), 1e-3), `exp[${i}]`);
      assert.ok(close(out2.data[i], -data[i], 1e-5), `neg[${i}]`);
    }
  });
});

describe('Chained matmuls (deep linear)', () => {
  it('5 matmuls in sequence', () => {
    const D = 8;
    const func = buildFunction('deep_linear', [
      T([4, D]), T([D, D]), T([D, D]), T([D, D]), T([D, D]), T([D, D]),
    ], [T([4, D])], (b, [x, w1, w2, w3, w4, w5]) => {
      let cur = b.matmul(x, w1).getResult(0);
      cur = b.matmul(cur, w2).getResult(0);
      cur = b.matmul(cur, w3).getResult(0);
      cur = b.matmul(cur, w4).getResult(0);
      cur = b.matmul(cur, w5).getResult(0);
      b.returnOp([cur]);
    });
    const compiled = compile(func);
    const out = RuntimeTensor.zeros([4, D]);
    const X = RuntimeTensor.fromArray(rand(4 * D), [4, D]);
    const Ws = Array.from({ length: 5 }, () => RuntimeTensor.fromArray(rand(D * D), [D, D]));
    compiled.run('deep_linear', X, ...Ws, out);
    assertFinite(out.data, 'deep_linear');
  });
});

describe('Mixed dtype pipeline', () => {
  it('f32 compute -> i32 convert -> f32 convert roundtrip', () => {
    const N = 8;
    const func = buildFunction('dtype_rt', [T([N])], [T([N])], (b, [x]) => {
      const added = b.add(x, b.broadcast(b.scalarConstant(0.5, f32).getResult(0), [N], []).getResult(0));
      const floored = b.floor(added.getResult(0));
      const asInt = b.convert(floored.getResult(0), i32);
      const back = b.convert(asInt.getResult(0), f32);
      b.returnOp([back.getResult(0)]);
    });
    const compiled = compile(func);
    const X = RuntimeTensor.fromArray([1.2, 2.7, -0.3, 3.9, 0.1, -1.6, 5.5, 4.4], [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('dtype_rt', X, out);
    for (let i = 0; i < N; i++) {
      const expected = Math.floor(X.data[i] + 0.5) | 0;
      assert.equal(out.data[i], expected, `dtype_rt[${i}]: ${out.data[i]} != ${expected}`);
    }
  });
});

describe('Stacked reductions', () => {
  it('sum rows then sum columns (two separate reduces)', () => {
    const R = 3, C = 4;
    const func = buildFunction('double_reduce', [T([R, C])], [T([])], (b, [x]) => {
      const init = b.scalarConstant(0, f32);
      const rowSum = b.reduce(x, init.getResult(0), [1], 'sum');
      const total = b.reduce(rowSum.getResult(0), init.getResult(0), [0], 'sum');
      b.returnOp([total.getResult(0)]);
    });
    const compiled = compile(func);
    const data = Float32Array.from({ length: R * C }, (_, i) => i + 1);
    const X = RuntimeTensor.fromArray(data, [R, C]);
    const out = RuntimeTensor.zeros([]);
    compiled.run('double_reduce', X, out);
    let expected = 0;
    for (let i = 0; i < data.length; i++) expected += data[i];
    assert.ok(close(out.data[0], expected), `total: ${out.data[0]} != ${expected}`);
  });
});

describe('GPU codegen stress', () => {
  it('transformer attention compiles to CUDA', () => {
    const B = 1, S = 4, D = 8;
    const func = buildFunction('gpu_attn', [T([B, S, D]), T([B, S, D]), T([B, S, D])], [T([B, S, D])],
      (b, [Q, K, V]) => {
        const scores = b.dot(Q, K, [2], [2], [0], [0]);
        const scale = b.broadcast(b.scalarConstant(1.0 / Math.sqrt(D), f32).getResult(0), [B, S, S], []);
        const attn = b.softmax(b.mul(scores.getResult(0), scale.getResult(0)).getResult(0), -1);
        b.returnOp([b.dot(attn.getResult(0), V, [2], [1], [0], [0]).getResult(0)]);
      }
    );
    const compiled = compileGPU(func);
    const source = compiled.getSource('gpu_attn');
    assert.ok(source.includes('__global__'));
    assert.ok(source.includes('expf'));
    assert.ok(source.length > 200, 'CUDA source should be substantial');
  });

  it('layernorm compiles to CUDA', () => {
    const func = buildFunction('gpu_ln', [T([4, 16]), T([16]), T([16])], [T([4, 16])],
      (b, [x, g, bt]) => { b.returnOp([b.layernorm(x, g, bt, -1, 1e-5).getResult(0)]); }
    );
    const compiled = compileGPU(func);
    assert.ok(compiled.getSource('gpu_ln').includes('__global__'));
  });
});

describe('Compare + select patterns', () => {
  it('leaky relu: where(x > 0, x, 0.01*x)', () => {
    const N = 16;
    const func = buildFunction('leaky_relu', [T([N])], [T([N])], (b, [x]) => {
      const zero = b.broadcast(b.scalarConstant(0, f32).getResult(0), [N], []);
      const alpha = b.broadcast(b.scalarConstant(0.01, f32).getResult(0), [N], []);
      const mask = b.compare(x, zero.getResult(0), 'gt');
      const scaled = b.mul(alpha.getResult(0), x);
      const out = b.select(mask.getResult(0), x, scaled.getResult(0));
      b.returnOp([out.getResult(0)]);
    });
    const compiled = compileFused(func);
    const data = rand(N);
    const X = RuntimeTensor.fromArray(data, [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('leaky_relu', X, out);
    for (let i = 0; i < N; i++) {
      const expected = data[i] > 0 ? data[i] : 0.01 * data[i];
      assert.ok(close(out.data[i], expected, 1e-4), `leaky[${i}]: ${out.data[i]} != ${expected}`);
    }
  });

  it('hard sigmoid: clamp(x * 0.2 + 0.5, 0, 1)', () => {
    const N = 16;
    const func = buildFunction('hard_sigmoid', [T([N])], [T([N])], (b, [x]) => {
      const scale = b.broadcast(b.scalarConstant(0.2, f32).getResult(0), [N], []);
      const shift = b.broadcast(b.scalarConstant(0.5, f32).getResult(0), [N], []);
      const lo = b.broadcast(b.scalarConstant(0, f32).getResult(0), [N], []);
      const hi = b.broadcast(b.scalarConstant(1, f32).getResult(0), [N], []);
      const linear = b.add(b.mul(x, scale.getResult(0)).getResult(0), shift.getResult(0));
      const out = b.clamp(lo.getResult(0), linear.getResult(0), hi.getResult(0));
      b.returnOp([out.getResult(0)]);
    });
    const compiled = compileFused(func);
    const data = Float32Array.from({ length: N }, (_, i) => (i - 8) * 0.8);
    const X = RuntimeTensor.fromArray(data, [N]);
    const out = RuntimeTensor.zeros([N]);
    compiled.run('hard_sigmoid', X, out);
    for (let i = 0; i < N; i++) {
      const expected = Math.min(1, Math.max(0, data[i] * 0.2 + 0.5));
      assert.ok(close(out.data[i], expected, 1e-4), `hsig[${i}]: ${out.data[i]} != ${expected}`);
    }
  });
});
