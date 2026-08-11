import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { countLoops, countTempBuffers } from '../../../_utils/kernel_source.js';
import { randomArray } from '../../../_utils/rng.js';
import * as ref from '../../../_utils/reference_ops.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

function expectMatches(actual, expected, label, tol = 1e-4) {
  const r = ref.expectClose(actual, expected, tol, label);
  expect(r.ok, r.message).toBe(true);
}

function run(result, name, inputs, outputShapes) {
  const inArrays = inputs.map(i =>
    i instanceof Float32Array ? i : new Float32Array(i)
  );
  const outArrays = outputShapes.map(s => {
    let n = 1;
    for (const d of s) n *= d;
    return new Float32Array(n);
  });
  result.run(name, ...inArrays, ...outArrays);
  return outArrays;
}

describe('LSTM cell: sigmoid gates + tanh + hadamard', () => {
  it('matches a reference LSTM cell: gate order i,f,g,o with the untouched tail carried through', () => {
    const xt = new TensorType([1, 4], ScalarType.F32);
    const ht = new TensorType([1, 4], ScalarType.F32);
    const ct = new TensorType([1, 4], ScalarType.F32);
    const wi = new TensorType([8, 4], ScalarType.F32);
    const bi = new TensorType([1, 4], ScalarType.F32);

    const func = buildFunction('lstm_cell',
      [xt, ht, ct, wi, bi],
      [ht, ct],
      (b, args) => {
        const xh = b.concat([args[0], args[1]], 1).getResult(0);
        const gates = b.matmul(xh, args[3]).getResult(0);
        const biased = b.add(gates, args[4]).getResult(0);

        const i_gate = b.sigmoid(b.slice(biased, [0, 0], [1, 1]).getResult(0)).getResult(0);
        const f_gate = b.sigmoid(b.slice(biased, [0, 1], [1, 2]).getResult(0)).getResult(0);
        const g_val  = b.tanh(b.slice(biased, [0, 2], [1, 3]).getResult(0)).getResult(0);
        const o_gate = b.sigmoid(b.slice(biased, [0, 3], [1, 4]).getResult(0)).getResult(0);

        const fc = b.mul(f_gate, b.slice(args[2], [0, 0], [1, 1]).getResult(0)).getResult(0);
        const ig = b.mul(i_gate, g_val).getResult(0);
        const newC = b.add(fc, ig).getResult(0);
        const newH = b.mul(o_gate, b.tanh(newC).getResult(0)).getResult(0);

        b.returnOp([
          b.concat([newH, b.slice(args[1], [0, 1], [1, 4]).getResult(0)], 1).getResult(0),
          b.concat([newC, b.slice(args[2], [0, 1], [1, 4]).getResult(0)], 1).getResult(0)
        ]);
      }
    );

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const x = randomArray(11, 4);
    const h = randomArray(12, 4);
    const c = randomArray(13, 4);
    const w = randomArray(14, 32, -0.5, 0.5);
    const bias = randomArray(15, 4, -0.2, 0.2);
    const [newH, newC] = run(r, 'lstm_cell', [x, h, c, w, bias], [[1, 4], [1, 4]]);

    const row = [1, 4];
    const biased = ref.add(ref.matmul(ref.concat([x, h], [row, row], 1), [1, 8], w, [8, 4]), bias);
    const gate = (lo, hi) => ref.slice(biased, row, [0, lo], [1, hi]);
    const iG = ref.sigmoid(gate(0, 1));
    const fG = ref.sigmoid(gate(1, 2));
    const gV = ref.tanh(gate(2, 3));
    const oG = ref.sigmoid(gate(3, 4));
    const expC = ref.add(ref.mul(fG, ref.slice(c, row, [0, 0], [1, 1])), ref.mul(iG, gV));
    const expH = ref.mul(oG, ref.tanh(expC));

    expectMatches(newH, ref.concat([expH, ref.slice(h, row, [0, 1], [1, 4])], [[1, 1], [1, 3]], 1), 'newH');
    expectMatches(newC, ref.concat([expC, ref.slice(c, row, [0, 1], [1, 4])], [[1, 1], [1, 3]], 1), 'newC');
  });
});

describe('Squeeze-and-Excitation block', () => {
  it('matches a reference SE block: global avg-pool, two FCs, and a per-channel sigmoid rescale', () => {
    const x     = new TensorType([1, 8, 4, 4], ScalarType.F32);
    const w1    = new TensorType([8, 2], ScalarType.F32);
    const w2    = new TensorType([2, 8], ScalarType.F32);
    const out   = new TensorType([1, 8, 4, 4], ScalarType.F32);

    const func = buildFunction('se_block', [x, w1, w2], [out], (b, args) => {
      const pooled = b.pool2d(args[0], 'avg', [4, 4], [4, 4], [[0, 0], [0, 0]]).getResult(0);
      const flat = b.reshape(pooled, [1, 8]).getResult(0);
      const fc1 = b.matmul(flat, args[1]).getResult(0);
      const act = b.relu(fc1).getResult(0);
      const fc2 = b.matmul(act, args[2]).getResult(0);
      const scale = b.sigmoid(fc2).getResult(0);
      const scaleBcast = b.broadcast(
        b.reshape(scale, [1, 8, 1, 1]).getResult(0),
        [1, 8, 4, 4],
        [0, 1, 2, 3]
      ).getResult(0);
      const scaled = b.mul(args[0], scaleBcast).getResult(0);
      b.returnOp([scaled]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(21, 128, 0.5, 2);
    const w1Data = randomArray(22, 16, -0.5, 0.5);
    const w2Data = randomArray(23, 16, -0.5, 0.5);
    const [result] = run(r, 'se_block', [xData, w1Data, w2Data], [[1, 8, 4, 4]]);

    const pooled = ref.reduce(xData, [1, 8, 4, 4], [2, 3], 'mean');
    const fc2 = ref.matmul(ref.relu(ref.matmul(pooled, [1, 8], w1Data, [8, 2])), [1, 2], w2Data, [2, 8]);
    const scale = ref.broadcastTo(ref.sigmoid(fc2), [1, 8, 1, 1], [1, 8, 4, 4]);
    expectMatches(result, ref.mul(xData, scale), 'se_block');
  });
});

describe('U-Net skip connection: conv -> concat with encoder feature', () => {
  it('matches a reference U-Net skip: channel concat feeding a 3x3 conv, with encoder channels first', () => {
    const enc  = new TensorType([1, 4, 4, 4], ScalarType.F32);
    const dec  = new TensorType([1, 8, 4, 4], ScalarType.F32);
    const k    = new TensorType([4, 12, 3, 3], ScalarType.F32);
    const out  = new TensorType([1, 4, 2, 2], ScalarType.F32);

    const func = buildFunction('unet_skip', [enc, dec, k], [out], (b, args) => {
      const cat = b.concat([args[0], args[1]], 1).getResult(0);
      const conv = b.conv(cat, args[2], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const act = b.relu(conv).getResult(0);
      b.returnOp([act]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const encData = randomArray(31, 64);
    const decData = randomArray(32, 128);
    const kData = randomArray(33, 4 * 12 * 9, -0.2, 0.2);
    const [result] = run(r, 'unet_skip', [encData, decData, kData], [[1, 4, 2, 2]]);

    const cat = ref.concat([encData, decData], [[1, 4, 4, 4], [1, 8, 4, 4]], 1);
    expectMatches(result, ref.relu(ref.conv2d(cat, [1, 12, 4, 4], kData, [4, 12, 3, 3]).data), 'unet_skip');
  });
});

describe('Cross-attention: Q from decoder, K/V from encoder', () => {
  it('compiles and produces valid attention output', () => {
    const Q   = new TensorType([4, 8], ScalarType.F32);
    const K   = new TensorType([6, 8], ScalarType.F32);
    const V   = new TensorType([6, 8], ScalarType.F32);
    const out = new TensorType([4, 8], ScalarType.F32);

    const func = buildFunction('cross_attn', [Q, K, V], [out], (b, args) => {
      const kt = b.transpose(args[1], [1, 0]).getResult(0);
      const scores = b.matmul(args[0], kt).getResult(0);
      const scaleVal = b.scalarConstant(1.0 / Math.sqrt(8), ScalarType.F32).getResult(0);
      const scaleBcast = b.broadcast(scaleVal, [4, 6], []).getResult(0);
      const scaled = b.mul(scores, scaleBcast).getResult(0);
      const weights = b.softmax(scaled).getResult(0);
      const attended = b.matmul(weights, args[2]).getResult(0);
      b.returnOp([attended]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const qData = new Float32Array(32).fill(0).map((_, i) => Math.sin(i * 0.1));
    const kData = new Float32Array(48).fill(0).map((_, i) => Math.cos(i * 0.1));
    const vData = new Float32Array(48).fill(1);
    const [result] = run(r, 'cross_attn', [qData, kData, vData], [[4, 8]]);
    expect(result.every(v => isFinite(v))).toBe(true);
    for (let i = 0; i < 32; i++) {
      expect(result[i]).toBeCloseTo(1.0, 1);
    }
  });
});

describe('MobileNet inverted residual: expand -> depthwise -> project', () => {
  it('matches a reference inverted bottleneck: 1x1 expand, depthwise 3x3 with groups=16, 1x1 project', () => {
    const x   = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const ke  = new TensorType([16, 4, 1, 1], ScalarType.F32);
    const kd  = new TensorType([16, 1, 3, 3], ScalarType.F32);
    const kp  = new TensorType([4, 16, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 4, 6, 6], ScalarType.F32);

    const func = buildFunction('inverted_res', [x, ke, kd, kp], [out], (b, args) => {
      const expanded = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const act1 = b.silu(expanded).getResult(0);
      const dw = b.conv(act1, args[2], [1, 1], [[0, 0], [0, 0]], { groups: 16 }).getResult(0);
      const act2 = b.silu(dw).getResult(0);
      const projected = b.conv(act2, args[3], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      b.returnOp([projected]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(41, 256);
    const keData = randomArray(42, 64, -0.4, 0.4);
    const kdData = randomArray(43, 144, -0.4, 0.4);
    const kpData = randomArray(44, 64, -0.4, 0.4);
    const [result] = run(r, 'inverted_res', [xData, keData, kdData, kpData], [[1, 4, 6, 6]]);

    const expanded = ref.silu(ref.conv2d(xData, [1, 4, 8, 8], keData, [16, 4, 1, 1]).data);
    const depthwise = ref.silu(ref.conv2d(expanded, [1, 16, 8, 8], kdData, [16, 1, 3, 3], { groups: 16 }).data);
    expectMatches(result, ref.conv2d(depthwise, [1, 16, 6, 6], kpData, [4, 16, 1, 1]).data, 'inverted_res');
  });
});

describe('Feature Pyramid Network: multi-scale feature fusion', () => {
  it('matches a reference FPN merge: 1x1 lateral conv, nearest upsample to 8x8, add the skip, relu', () => {
    const c3  = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const c4  = new TensorType([1, 8, 4, 4], ScalarType.F32);
    const lat = new TensorType([4, 8, 1, 1], ScalarType.F32);
    const p3  = new TensorType([1, 4, 8, 8], ScalarType.F32);

    const func = buildFunction('fpn_merge', [c3, c4, lat], [p3], (b, args) => {
      const lateral = b.conv(args[1], args[2], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const upsampled = b.resize(lateral, [8, 8], 'nearest').getResult(0);
      const merged = b.add(args[0], upsampled).getResult(0);
      const out = b.relu(merged).getResult(0);
      b.returnOp([out]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const c3Data = randomArray(51, 256);
    const c4Data = randomArray(52, 128);
    const latData = randomArray(53, 32, -0.5, 0.5);
    const [result] = run(r, 'fpn_merge', [c3Data, c4Data, latData], [[1, 4, 8, 8]]);

    const lateral = ref.conv2d(c4Data, [1, 8, 4, 4], latData, [4, 8, 1, 1]).data;
    const upsampled = ref.resizeNearest(lateral, [1, 4, 4, 4], [8, 8]);
    expectMatches(result, ref.relu(ref.add(c3Data, upsampled)), 'fpn_merge');
  });
});

describe('Transformer encoder block — full', () => {
  it('matches a reference encoder block: scaled dot-product attention, both residuals, and both layernorms with their own gamma/beta', () => {
    const x     = new TensorType([4, 16], ScalarType.F32);
    const wq    = new TensorType([16, 16], ScalarType.F32);
    const wk    = new TensorType([16, 16], ScalarType.F32);
    const wv    = new TensorType([16, 16], ScalarType.F32);
    const wo    = new TensorType([16, 16], ScalarType.F32);
    const g1    = new TensorType([16], ScalarType.F32);
    const b1    = new TensorType([16], ScalarType.F32);
    const wff1  = new TensorType([16, 64], ScalarType.F32);
    const wff2  = new TensorType([64, 16], ScalarType.F32);
    const g2    = new TensorType([16], ScalarType.F32);
    const b2    = new TensorType([16], ScalarType.F32);
    const out   = new TensorType([4, 16], ScalarType.F32);

    const func = buildFunction('encoder_block',
      [x, wq, wk, wv, wo, g1, b1, wff1, wff2, g2, b2],
      [out],
      (b, args) => {
        const q = b.matmul(args[0], args[1]).getResult(0);
        const k = b.matmul(args[0], args[2]).getResult(0);
        const v = b.matmul(args[0], args[3]).getResult(0);
        const kt = b.transpose(k, [1, 0]).getResult(0);
        const scores = b.matmul(q, kt).getResult(0);
        const scaleVal = b.scalarConstant(1 / Math.sqrt(16), ScalarType.F32).getResult(0);
        const scaleBc = b.broadcast(scaleVal, [4, 4], []).getResult(0);
        const scaled = b.mul(scores, scaleBc).getResult(0);
        const attn = b.softmax(scaled).getResult(0);
        const ctx = b.matmul(attn, v).getResult(0);
        const proj = b.matmul(ctx, args[4]).getResult(0);
        const res1 = b.add(args[0], proj).getResult(0);
        const norm1 = b.layernorm(res1, args[5], args[6]).getResult(0);

        const ff1 = b.matmul(norm1, args[7]).getResult(0);
        const act = b.gelu(ff1).getResult(0);
        const ff2 = b.matmul(act, args[8]).getResult(0);
        const res2 = b.add(norm1, ff2).getResult(0);
        const norm2 = b.layernorm(res2, args[9], args[10]).getResult(0);
        b.returnOp([norm2]);
      }
    );

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const p = {
      x: randomArray(61, 64),
      wq: randomArray(62, 256, -0.3, 0.3),
      wk: randomArray(63, 256, -0.3, 0.3),
      wv: randomArray(64, 256, -0.3, 0.3),
      wo: randomArray(65, 256, -0.3, 0.3),
      g1: randomArray(66, 16, 0.5, 1.5),
      b1: randomArray(67, 16, -0.3, 0.3),
      wff1: randomArray(68, 1024, -0.2, 0.2),
      wff2: randomArray(69, 1024, -0.2, 0.2),
      g2: randomArray(70, 16, 0.5, 1.5),
      b2: randomArray(71, 16, -0.3, 0.3),
    };
    const [result] = run(r, 'encoder_block',
      [p.x, p.wq, p.wk, p.wv, p.wo, p.g1, p.b1, p.wff1, p.wff2, p.g2, p.b2],
      [[4, 16]]);

    const expected = ref.transformerEncoderBlock(p.x, { seq: 4, dModel: 16, dFF: 64, ...p });
    expectMatches(result, expected, 'encoder_block', 1e-3);
  });
});

describe('Dilated conv tower: multi-rate receptive field', () => {
  it('matches a reference dilated tower: dilation-2 padded conv then a stride-2 conv', () => {
    const x  = new TensorType([1, 1, 16, 16], ScalarType.F32);
    const k1 = new TensorType([4, 1, 3, 3], ScalarType.F32);
    const k2 = new TensorType([4, 4, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 4, 6, 6], ScalarType.F32);

    const func = buildFunction('dilated_tower', [x, k1, k2], [out], (b, args) => {
      const c1 = b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]], { dilation: [2, 2] }).getResult(0);
      const a1 = b.relu(c1).getResult(0);
      const c2 = b.conv(a1, args[2], [2, 2], [[0, 0], [0, 0]]).getResult(0);
      const a2 = b.relu(c2).getResult(0);
      b.returnOp([a2]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(81, 256);
    const k1Data = randomArray(82, 36, -0.3, 0.3);
    const k2Data = randomArray(83, 576, -0.15, 0.15);
    const [result] = run(r, 'dilated_tower', [xData, k1Data, k2Data], [[1, 4, 6, 6]]);

    const c1 = ref.relu(ref.conv2d(xData, [1, 1, 16, 16], k1Data, [4, 1, 3, 3], { pad: 1, dilation: 2 }).data);
    expectMatches(result, ref.relu(ref.conv2d(c1, [1, 4, 14, 14], k2Data, [4, 4, 3, 3], { stride: 2 }).data), 'dilated_tower');
  });
});

describe('Batch matmul: batched attention scores', () => {
  it('compiles and verifies batched matmul correctness', () => {
    const q = new TensorType([2, 4, 8], ScalarType.F32);
    const k = new TensorType([2, 8, 4], ScalarType.F32);
    const out = new TensorType([2, 4, 4], ScalarType.F32);

    const func = buildFunction('batch_mm', [q, k], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const qData = new Float32Array(64).fill(0);
    const kData = new Float32Array(64).fill(0);
    for (let b = 0; b < 2; b++) {
      for (let i = 0; i < 4; i++) {
        qData[b * 32 + i * 8 + i] = 1;
        kData[b * 32 + i * 4 + i] = 1;
      }
    }
    const [result] = run(r, 'batch_mm', [qData, kData], [[2, 4, 4]]);
    for (let b = 0; b < 2; b++) {
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          const expected = i === j ? 1 : 0;
          expect(result[b * 16 + i * 4 + j]).toBeCloseTo(expected, 5);
        }
      }
    }
  });
});

describe('MLP-Mixer style: transpose -> matmul -> transpose (token mixing)', () => {
  it('matches a reference token-mixing layer: mixing happens across tokens, not channels', () => {
    const x  = new TensorType([8, 16], ScalarType.F32);
    const w  = new TensorType([8, 8], ScalarType.F32);
    const out = new TensorType([8, 16], ScalarType.F32);

    const func = buildFunction('token_mix', [x, w], [out], (b, args) => {
      const xt = b.transpose(args[0], [1, 0]).getResult(0);
      const mixed = b.matmul(xt, args[1]).getResult(0);
      const back = b.transpose(mixed, [1, 0]).getResult(0);
      const act = b.gelu(back).getResult(0);
      b.returnOp([act]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(91, 128);
    const wData = randomArray(92, 64, -0.6, 0.6);
    const [result] = run(r, 'token_mix', [xData, wData], [[8, 16]]);

    const mixed = ref.matmul(ref.transpose(xData, [8, 16], [1, 0]), [16, 8], wData, [8, 8]);
    expectMatches(result, ref.gelu(ref.transpose(mixed, [16, 8], [1, 0])), 'token_mix');
  });
});

describe('Type conversion pipeline: f32 -> clamp -> convert', () => {
  it('clamp + type coercion produces bounded output', () => {
    const x   = new TensorType([16], ScalarType.F32);
    const out = new TensorType([16], ScalarType.F32);

    const func = buildFunction('clamp_pipe', [x], [out], (b, args) => {
      const lo = b.scalarConstant(0, ScalarType.F32).getResult(0);
      const hi = b.scalarConstant(1, ScalarType.F32).getResult(0);
      const loBcast = b.broadcast(lo, [16], []).getResult(0);
      const hiBcast = b.broadcast(hi, [16], []).getResult(0);
      const clamped = b.clamp(loBcast, args[0], hiBcast).getResult(0);
      b.returnOp([clamped]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array([-2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, -3, 0.1, 0.9, -0.1, 1.1]);
    const [result] = run(r, 'clamp_pipe', [xData], [[16]]);
    for (let i = 0; i < 16; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
      expect(result[i]).toBeLessThanOrEqual(1);
    }
    expect(result[3]).toBe(0);
    expect(result[7]).toBe(1);
    expect(result[4]).toBeCloseTo(0.25);
  });
});

describe('Cosine similarity: dot / (norm * norm)', () => {
  it('computes normalized dot product', () => {
    const a = new TensorType([8], ScalarType.F32);
    const b_ = new TensorType([8], ScalarType.F32);
    const out = new TensorType([1], ScalarType.F32);

    const func = buildFunction('cosine_sim', [a, b_], [out], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32).getResult(0);

      const ab = b.mul(args[0], args[1]).getResult(0);
      const dotProd = b.reduce(ab, zero, [0], 'sum').getResult(0);

      const a2 = b.mul(args[0], args[0]).getResult(0);
      const normA2 = b.reduce(a2, zero, [0], 'sum').getResult(0);
      const normA = b.sqrt(normA2).getResult(0);

      const b2 = b.mul(args[1], args[1]).getResult(0);
      const normB2 = b.reduce(b2, zero, [0], 'sum').getResult(0);
      const normB = b.sqrt(normB2).getResult(0);

      const denom = b.mul(normA, normB).getResult(0);
      const sim = b.div(dotProd, denom).getResult(0);
      const reshaped = b.reshape(sim, [1]).getResult(0);
      b.returnOp([reshaped]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const aData = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const bData = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const [result] = run(r, 'cosine_sim', [aData, bData], [[1]]);
    expect(result[0]).toBeCloseTo(1.0, 5);

    const cData = new Float32Array([0, 1, 0, 0, 0, 0, 0, 0]);
    const [result2] = run(r, 'cosine_sim', [aData, cData], [[1]]);
    expect(result2[0]).toBeCloseTo(0.0, 5);
  });
});

describe('Deep residual chain: 4 residual additions', () => {
  it('maintains precision through deep chain', () => {
    const t = new TensorType([4, 8], ScalarType.F32);

    const func = buildFunction('deep_res', [t, t, t, t, t], [t], (b, args) => {
      let v = args[0];
      for (let i = 1; i <= 4; i++) {
        const transformed = b.tanh(b.add(v, args[i]).getResult(0)).getResult(0);
        v = b.add(v, transformed).getResult(0);
      }
      b.returnOp([v]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const inputs = Array.from({ length: 5 }, () => new Float32Array(32).fill(0.1));
    const [result] = run(r, 'deep_res', inputs, [[4, 8]]);
    expect(result.every(v => isFinite(v))).toBe(true);
    expect(result[0]).toBeGreaterThan(0.1);
  });
});

describe('Grouped convolution: group=2', () => {
  it('matches a reference groups=2 conv, so each filter sees only its own channel half', () => {
    const x   = new TensorType([1, 4, 4, 4], ScalarType.F32);
    const k   = new TensorType([4, 2, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 4, 2, 2], ScalarType.F32);

    const func = buildFunction('group_conv', [x, k], [out], (b, args) => {
      const conv = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]], { groups: 2 }).getResult(0);
      const act = b.relu(conv).getResult(0);
      b.returnOp([act]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(101, 64);
    const kData = randomArray(102, 72, -0.5, 0.5);
    const [result] = run(r, 'group_conv', [xData, kData], [[1, 4, 2, 2]]);

    expectMatches(result, ref.relu(ref.conv2d(xData, [1, 4, 4, 4], kData, [4, 2, 3, 3], { groups: 2 }).data), 'group_conv');
  });
});

describe('Detection head: conv -> reshape -> transpose (anchor format)', () => {
  it('compiles and reshapes conv output to [batch, anchors, classes]', () => {
    const x   = new TensorType([1, 8, 4, 4], ScalarType.F32);
    const k   = new TensorType([12, 8, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 16, 12], ScalarType.F32);

    const func = buildFunction('det_head', [x, k], [out], (b, args) => {
      const conv = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const reshaped = b.reshape(conv, [1, 12, 16]).getResult(0);
      const transposed = b.transpose(reshaped, [0, 2, 1]).getResult(0);
      const probs = b.sigmoid(transposed).getResult(0);
      b.returnOp([probs]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array(128).fill(0.5);
    const kData = new Float32Array(96).fill(0.1);
    const [result] = run(r, 'det_head', [xData, kData], [[1, 16, 12]]);
    expect(result.every(v => v > 0 && v < 1)).toBe(true);
  });
});

describe('Gradient-like: y = x^2, dy/dx = 2x via explicit graph', () => {
  it('forward and manual backward produce correct gradients', () => {
    const x    = new TensorType([8], ScalarType.F32);
    const out  = new TensorType([8], ScalarType.F32);
    const grad = new TensorType([8], ScalarType.F32);

    const fwd = buildFunction('fwd_x2', [x], [out], (b, args) => {
      b.returnOp([b.mul(args[0], args[0]).getResult(0)]);
    });

    const bwd = buildFunction('bwd_x2', [x, grad], [out], (b, args) => {
      const two = b.scalarConstant(2, ScalarType.F32).getResult(0);
      const twoBc = b.broadcast(two, [8], []).getResult(0);
      const dx = b.mul(b.mul(twoBc, args[0]).getResult(0), args[1]).getResult(0);
      b.returnOp([dx]);
    });

    const rf = compile(fwd);
    const rb = compile(bwd);
    expect(rf.succeeded).toBe(true);
    expect(rb.succeeded).toBe(true);

    const xData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const [y] = run(rf, 'fwd_x2', [xData], [[8]]);
    expect(Array.from(y)).toEqual([1, 4, 9, 16, 25, 36, 49, 64]);

    const gradOut = new Float32Array(8).fill(1);
    const [dx] = run(rb, 'bwd_x2', [xData, gradOut], [[8]]);
    expect(Array.from(dx)).toEqual([2, 4, 6, 8, 10, 12, 14, 16]);
  });
});

describe('KERNEL QUALITY — production patterns', () => {
  it('elementwise broadcast fusion: no temp buffers for bias + activation', () => {
    const x  = new TensorType([4, 16], ScalarType.F32);
    const bi = new TensorType([16], ScalarType.F32);
    const out = new TensorType([4, 16], ScalarType.F32);

    const func = buildFunction('bias_act', [x, bi], [out], (b, args) => {
      const bcast = b.broadcast(args[1], [4, 16], [1]).getResult(0);
      const biased = b.add(args[0], bcast).getResult(0);
      const act = b.silu(biased).getResult(0);
      b.returnOp([act]);
    });

    const r = compile(func);
    const src = r.getSource('bias_act');
    expect(countTempBuffers(src)).toBe(0);
  });

  it('chained scalar broadcasts all inlined', () => {
    const t = new TensorType([32], ScalarType.F32);

    const func = buildFunction('multi_scalar', [t], [t], (b, args) => {
      const s1 = b.broadcast(b.scalarConstant(0.5, ScalarType.F32).getResult(0), [32], []).getResult(0);
      const s2 = b.broadcast(b.scalarConstant(2.0, ScalarType.F32).getResult(0), [32], []).getResult(0);
      const s3 = b.broadcast(b.scalarConstant(-1, ScalarType.F32).getResult(0), [32], []).getResult(0);
      let v = b.mul(args[0], s1).getResult(0);
      v = b.add(v, s2).getResult(0);
      v = b.mul(v, s3).getResult(0);
      b.returnOp([v]);
    });

    const r = compile(func);
    const src = r.getSource('multi_scalar');
    expect(countTempBuffers(src)).toBe(0);
    expect(countLoops(src)).toBe(1);

    const [result] = run(r, 'multi_scalar', [[1, 2, 3, 4, 5, 6, 7, 8,
      9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]], [[32]]);
    expect(result[0]).toBeCloseTo(-(0.5 * 1 + 2));
    expect(result[1]).toBeCloseTo(-(0.5 * 2 + 2));
  });

  it('no arithmetic noise in 1x1 pointwise conv', () => {
    const x   = new TensorType([1, 16, 4, 4], ScalarType.F32);
    const k   = new TensorType([32, 16, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 32, 4, 4], ScalarType.F32);

    const func = buildFunction('pw_conv', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('pw_conv');
    expect(src).not.toMatch(/\+\s*0\b/);
    expect(src).not.toMatch(/\b0\s*\+/);
    expect(src).not.toMatch(/\b0\s*\*/);
    expect(src).not.toMatch(/\*\s*0\b/);
    expect(src).not.toMatch(/\*\s*1\b/);
  });

  it('no bounds checks in zero-padded conv', () => {
    const x   = new TensorType([1, 16, 32, 32], ScalarType.F32);
    const k   = new TensorType([32, 16, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 32, 30, 30], ScalarType.F32);

    const func = buildFunction('conv_nopad_lg', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_nopad_lg');
    expect(src).not.toMatch(/>=\s*0/);
    expect(src).not.toMatch(/<\s*\d+\s*\)/);
  });

  it('5-op elementwise chain fully fused', () => {
    const t = new TensorType([64], ScalarType.F32);

    const func = buildFunction('chain5', [t, t], [t], (b, args) => {
      let v = b.add(args[0], args[1]).getResult(0);
      v = b.sigmoid(v).getResult(0);
      v = b.mul(v, args[0]).getResult(0);
      v = b.tanh(v).getResult(0);
      v = b.abs(v).getResult(0);
      b.returnOp([v]);
    });

    const r = compile(func);
    const src = r.getSource('chain5');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
  });

  it('reduction followed by broadcast and elementwise — minimal buffers', () => {
    const x = new TensorType([4, 8], ScalarType.F32);
    const out = new TensorType([4, 8], ScalarType.F32);

    const func = buildFunction('center', [x], [out], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32).getResult(0);
      const mean = b.reduce(args[0], zero, [1], 'mean').getResult(0);
      const meanBcast = b.broadcast(mean, [4, 8], [0]).getResult(0);
      const centered = b.sub(args[0], meanBcast).getResult(0);
      b.returnOp([centered]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 8,
      2, 4, 6, 8, 10, 12, 14, 16,
      0, 0, 0, 0, 0, 0, 0, 0,
      -1, -1, -1, -1, 1, 1, 1, 1
    ]);
    const [result] = run(r, 'center', [xData], [[4, 8]]);
    for (let row = 0; row < 4; row++) {
      let rowSum = 0;
      for (let col = 0; col < 8; col++) rowSum += result[row * 8 + col];
      expect(rowSum).toBeCloseTo(0, 3);
    }
  });

  it('identity reshape generates no modulo or division', () => {
    const t = new TensorType([8, 16], ScalarType.F32);
    const func = buildFunction('id_resh_lg', [t], [t], (b, args) => {
      b.returnOp([b.reshape(args[0], [8, 16]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('id_resh_lg');
    expect(src).not.toMatch(/%/);
    expect(src).not.toMatch(/Math\.trunc/);
  });
});

describe('Numerical stability — edge cases', () => {
  it('softmax with large inputs does not produce NaN/Inf', () => {
    const t = new TensorType([1, 8], ScalarType.F32);

    const func = buildFunction('sm_stable', [t], [t], (b, args) => {
      b.returnOp([b.softmax(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const xData = new Float32Array([100, 200, 300, 400, 500, 600, 700, 800]);
    const [result] = run(r, 'sm_stable', [xData], [[1, 8]]);
    expect(result.every(v => isFinite(v) && v >= 0)).toBe(true);
    let sum = 0;
    for (let i = 0; i < 8; i++) sum += result[i];
    expect(sum).toBeCloseTo(1.0, 3);
    expect(result[7]).toBeCloseTo(1.0, 3);
  });

  it('layernorm with uniform input produces zero-centered output', () => {
    const x     = new TensorType([2, 4], ScalarType.F32);
    const gamma = new TensorType([4], ScalarType.F32);
    const beta  = new TensorType([4], ScalarType.F32);

    const func = buildFunction('ln_uniform', [x, gamma, beta], [x], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2]).getResult(0)]);
    });

    const r = compile(func);
    const xData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const gamma_ = new Float32Array([1, 1, 1, 1]);
    const beta_ = new Float32Array([0, 0, 0, 0]);
    const [result] = run(r, 'ln_uniform', [xData, gamma_, beta_], [[2, 4]]);
    expect(result.every(v => isFinite(v))).toBe(true);
    for (let row = 0; row < 2; row++) {
      let rowSum = 0;
      for (let col = 0; col < 4; col++) rowSum += result[row * 4 + col];
      expect(rowSum).toBeCloseTo(0, 3);
    }
  });

  it('sigmoid at extreme inputs saturates correctly', () => {
    const t = new TensorType([4], ScalarType.F32);

    const func = buildFunction('sig_sat', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const xData = new Float32Array([-100, -10, 10, 100]);
    const [result] = run(r, 'sig_sat', [xData], [[4]]);
    expect(result[0]).toBeCloseTo(0, 5);
    expect(result[1]).toBeCloseTo(0, 3);
    expect(result[2]).toBeCloseTo(1, 3);
    expect(result[3]).toBeCloseTo(1, 5);
  });

  it('exp with negative input does not underflow to exactly zero', () => {
    const t = new TensorType([4], ScalarType.F32);

    const func = buildFunction('exp_neg', [t], [t], (b, args) => {
      b.returnOp([b.exp(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const xData = new Float32Array([-1, -5, -10, -20]);
    const [result] = run(r, 'exp_neg', [xData], [[4]]);
    for (let i = 0; i < 4; i++) {
      expect(result[i]).toBeGreaterThan(0);
      expect(result[i]).toBeCloseTo(Math.exp(xData[i]), 4);
    }
  });
});

describe('Complex data flow — diamond and fan patterns', () => {
  it('diamond: A feeds B and C, both feed D', () => {
    const t = new TensorType([16], ScalarType.F32);

    const func = buildFunction('diamond', [t], [t], (b, args) => {
      const a = b.exp(args[0]).getResult(0);
      const branch1 = b.sigmoid(a).getResult(0);
      const branch2 = b.tanh(a).getResult(0);
      const merged = b.mul(branch1, branch2).getResult(0);
      b.returnOp([merged]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);
    const src = r.getSource('diamond');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);

    const xData = new Float32Array(16).fill(0);
    const [result] = run(r, 'diamond', [xData], [[16]]);
    const expVal = Math.exp(0);
    const expected = (1 / (1 + Math.exp(-expVal))) * Math.tanh(expVal);
    for (let i = 0; i < 16; i++) {
      expect(result[i]).toBeCloseTo(expected, 5);
    }
  });

  it('fan-out: single input feeds 3 independent outputs', () => {
    const t = new TensorType([8], ScalarType.F32);

    const func = buildFunction('fan_out', [t], [t, t, t], (b, args) => {
      const a = b.sigmoid(args[0]).getResult(0);
      const b1 = b.tanh(args[0]).getResult(0);
      const c = b.neg(args[0]).getResult(0);
      b.returnOp([a, b1, c]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array([0, 1, -1, 2, -2, 0.5, -0.5, 3]);
    const [sigOut, tanhOut, negOut] = run(r, 'fan_out', [xData], [[8], [8], [8]]);
    for (let i = 0; i < 8; i++) {
      expect(sigOut[i]).toBeCloseTo(1 / (1 + Math.exp(-xData[i])), 4);
      expect(tanhOut[i]).toBeCloseTo(Math.tanh(xData[i]), 4);
      expect(negOut[i]).toBeCloseTo(-xData[i], 5);
    }
  });

  it('wide fan-in: 4 inputs merged via elementwise chain', () => {
    const t = new TensorType([16], ScalarType.F32);

    const func = buildFunction('fan_in', [t, t, t, t], [t], (b, args) => {
      const ab = b.add(args[0], args[1]).getResult(0);
      const cd = b.mul(args[2], args[3]).getResult(0);
      const merged = b.sub(ab, cd).getResult(0);
      const final = b.tanh(merged).getResult(0);
      b.returnOp([final]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);
    const src = r.getSource('fan_in');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);

    const inputs = [
      new Float32Array(16).fill(1),
      new Float32Array(16).fill(2),
      new Float32Array(16).fill(0.5),
      new Float32Array(16).fill(0.5),
    ];
    const [result] = run(r, 'fan_in', inputs, [[16]]);
    const expected = Math.tanh((1 + 2) - (0.5 * 0.5));
    for (let i = 0; i < 16; i++) {
      expect(result[i]).toBeCloseTo(expected, 5);
    }
  });
});
