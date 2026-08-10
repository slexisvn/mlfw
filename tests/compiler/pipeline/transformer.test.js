import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { randomArray } from '../../_utils/rng.js';
import * as ref from '../../_utils/reference_ops.js';

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget(), opts);
}

function expectMatches(actual, expected, label, tol = 1e-3) {
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

const SEQ = 4;
const D_MODEL = 8;
const N_HEADS = 2;
const D_HEAD = D_MODEL / N_HEADS;
const D_FF = 16;

const DIMS = { seq: SEQ, dModel: D_MODEL, dFF: D_FF };

function encoderParams(seed) {
  return {
    x: randomArray(seed, SEQ * D_MODEL),
    wq: randomArray(seed + 1, D_MODEL * D_MODEL, -0.4, 0.4),
    wk: randomArray(seed + 2, D_MODEL * D_MODEL, -0.4, 0.4),
    wv: randomArray(seed + 3, D_MODEL * D_MODEL, -0.4, 0.4),
    wo: randomArray(seed + 4, D_MODEL * D_MODEL, -0.4, 0.4),
    g1: randomArray(seed + 5, D_MODEL, 0.5, 1.5),
    b1: randomArray(seed + 6, D_MODEL, -0.3, 0.3),
    wff1: randomArray(seed + 7, D_MODEL * D_FF, -0.3, 0.3),
    wff2: randomArray(seed + 8, D_FF * D_MODEL, -0.3, 0.3),
    g2: randomArray(seed + 9, D_MODEL, 0.5, 1.5),
    b2: randomArray(seed + 10, D_MODEL, -0.3, 0.3),
  };
}

describe('Transformer — IR builder', () => {
  it('compiles scaled dot-product attention with multi-head reshape', () => {
    const q = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const k = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const v = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const out = new TensorType([SEQ, D_MODEL], ScalarType.F32);

    const func = buildFunction('mha_sdpa', [q, k, v], [out], (b, args) => {
      const qr = b.reshape(args[0], [SEQ, N_HEADS, D_HEAD]).getResult(0);
      const qt = b.transpose(qr, [1, 0, 2]).getResult(0);
      const kr = b.reshape(args[1], [SEQ, N_HEADS, D_HEAD]).getResult(0);
      const kt = b.transpose(kr, [1, 2, 0]).getResult(0);
      const vr = b.reshape(args[2], [SEQ, N_HEADS, D_HEAD]).getResult(0);
      const vt = b.transpose(vr, [1, 0, 2]).getResult(0);

      const scores = b.matmul(qt, kt).getResult(0);
      const scaleVal = b.scalarConstant(1 / Math.sqrt(D_HEAD), ScalarType.F32).getResult(0);
      const scaleBc = b.broadcast(scaleVal, [N_HEADS, SEQ, SEQ], []).getResult(0);
      const scaled = b.mul(scores, scaleBc).getResult(0);
      const attn = b.softmax(scaled).getResult(0);
      const ctx = b.matmul(attn, vt).getResult(0);

      const ct = b.transpose(ctx, [1, 0, 2]).getResult(0);
      const merged = b.reshape(ct, [SEQ, D_MODEL]).getResult(0);
      b.returnOp([merged]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const qData = new Float32Array(SEQ * D_MODEL).fill(0).map((_, i) => Math.sin(i * 0.3));
    const kData = new Float32Array(SEQ * D_MODEL).fill(0).map((_, i) => Math.cos(i * 0.3));
    const vData = new Float32Array(SEQ * D_MODEL).fill(1);
    const [result] = run(r, 'mha_sdpa', [qData, kData, vData], [[SEQ, D_MODEL]]);
    expect(result.every(v => isFinite(v))).toBe(true);
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBeCloseTo(1.0, 1);
    }
  });

  it('matches a reference encoder block with distinct Q/K/V/O projections and per-layernorm gamma/beta', () => {
    const x    = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const wq   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wk   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wv   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wo   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const g1   = new TensorType([D_MODEL], ScalarType.F32);
    const b1   = new TensorType([D_MODEL], ScalarType.F32);
    const wff1 = new TensorType([D_MODEL, D_FF], ScalarType.F32);
    const wff2 = new TensorType([D_FF, D_MODEL], ScalarType.F32);
    const g2   = new TensorType([D_MODEL], ScalarType.F32);
    const b2   = new TensorType([D_MODEL], ScalarType.F32);
    const out  = new TensorType([SEQ, D_MODEL], ScalarType.F32);

    const func = buildFunction('encoder_block',
      [x, wq, wk, wv, wo, g1, b1, wff1, wff2, g2, b2],
      [out],
      (b, args) => {
        const q = b.matmul(args[0], args[1]).getResult(0);
        const k = b.matmul(args[0], args[2]).getResult(0);
        const v = b.matmul(args[0], args[3]).getResult(0);
        const kt = b.transpose(k, [1, 0]).getResult(0);
        const scores = b.matmul(q, kt).getResult(0);
        const scaleVal = b.scalarConstant(1 / Math.sqrt(D_MODEL), ScalarType.F32).getResult(0);
        const scaleBc = b.broadcast(scaleVal, [SEQ, SEQ], []).getResult(0);
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

    const p = encoderParams(200);
    const [result] = run(r, 'encoder_block',
      [p.x, p.wq, p.wk, p.wv, p.wo, p.g1, p.b1, p.wff1, p.wff2, p.g2, p.b2],
      [[SEQ, D_MODEL]]);

    expectMatches(result, ref.transformerEncoderBlock(p.x, { ...DIMS, ...p }), 'encoder_block');
  });

  it('matches two reference encoder layers applied in sequence, each with its own weights', () => {
    const x    = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const wq1  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wk1  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wv1  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wo1  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const g1a  = new TensorType([D_MODEL], ScalarType.F32);
    const b1a  = new TensorType([D_MODEL], ScalarType.F32);
    const wf1a = new TensorType([D_MODEL, D_FF], ScalarType.F32);
    const wf1b = new TensorType([D_FF, D_MODEL], ScalarType.F32);
    const g1b  = new TensorType([D_MODEL], ScalarType.F32);
    const b1b  = new TensorType([D_MODEL], ScalarType.F32);
    const wq2  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wk2  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wv2  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wo2  = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const g2a  = new TensorType([D_MODEL], ScalarType.F32);
    const b2a  = new TensorType([D_MODEL], ScalarType.F32);
    const wf2a = new TensorType([D_MODEL, D_FF], ScalarType.F32);
    const wf2b = new TensorType([D_FF, D_MODEL], ScalarType.F32);
    const g2b  = new TensorType([D_MODEL], ScalarType.F32);
    const b2b  = new TensorType([D_MODEL], ScalarType.F32);
    const out  = new TensorType([SEQ, D_MODEL], ScalarType.F32);

    const func = buildFunction('stacked_encoder',
      [x, wq1, wk1, wv1, wo1, g1a, b1a, wf1a, wf1b, g1b, b1b,
        wq2, wk2, wv2, wo2, g2a, b2a, wf2a, wf2b, g2b, b2b],
      [out],
      (b, args) => {
        function encoderLayer(input, offset) {
          const q = b.matmul(input, args[offset]).getResult(0);
          const k = b.matmul(input, args[offset + 1]).getResult(0);
          const v = b.matmul(input, args[offset + 2]).getResult(0);
          const kt = b.transpose(k, [1, 0]).getResult(0);
          const scores = b.matmul(q, kt).getResult(0);
          const sv = b.scalarConstant(1 / Math.sqrt(D_MODEL), ScalarType.F32).getResult(0);
          const sbc = b.broadcast(sv, [SEQ, SEQ], []).getResult(0);
          const scaled = b.mul(scores, sbc).getResult(0);
          const attn = b.softmax(scaled).getResult(0);
          const ctx = b.matmul(attn, v).getResult(0);
          const proj = b.matmul(ctx, args[offset + 3]).getResult(0);
          const res1 = b.add(input, proj).getResult(0);
          const norm1 = b.layernorm(res1, args[offset + 4], args[offset + 5]).getResult(0);
          const ff1 = b.matmul(norm1, args[offset + 6]).getResult(0);
          const act = b.gelu(ff1).getResult(0);
          const ff2 = b.matmul(act, args[offset + 7]).getResult(0);
          const res2 = b.add(norm1, ff2).getResult(0);
          return b.layernorm(res2, args[offset + 8], args[offset + 9]).getResult(0);
        }

        const h1 = encoderLayer(args[0], 1);
        const h2 = encoderLayer(h1, 11);
        b.returnOp([h2]);
      }
    );

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const layer1 = encoderParams(300);
    const layer2 = encoderParams(400);
    const weightsOf = (p) => [p.wq, p.wk, p.wv, p.wo, p.g1, p.b1, p.wff1, p.wff2, p.g2, p.b2];
    const [result] = run(r, 'stacked_encoder',
      [layer1.x, ...weightsOf(layer1), ...weightsOf(layer2)],
      [[SEQ, D_MODEL]]);

    const h1 = ref.transformerEncoderBlock(layer1.x, { ...DIMS, ...layer1 });
    expectMatches(result, ref.transformerEncoderBlock(h1, { ...DIMS, ...layer2 }), 'stacked_encoder');
  });

  it('matches a reference decoder block where the causal mask zeroes attention to future positions', () => {
    const x    = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const wq   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wk   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wv   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const wo   = new TensorType([D_MODEL, D_MODEL], ScalarType.F32);
    const mask = new TensorType([SEQ, SEQ], ScalarType.F32);
    const g1   = new TensorType([D_MODEL], ScalarType.F32);
    const b1t  = new TensorType([D_MODEL], ScalarType.F32);
    const wff1 = new TensorType([D_MODEL, D_FF], ScalarType.F32);
    const wff2 = new TensorType([D_FF, D_MODEL], ScalarType.F32);
    const g2   = new TensorType([D_MODEL], ScalarType.F32);
    const b2t  = new TensorType([D_MODEL], ScalarType.F32);
    const out  = new TensorType([SEQ, D_MODEL], ScalarType.F32);

    const func = buildFunction('decoder_block',
      [x, wq, wk, wv, wo, mask, g1, b1t, wff1, wff2, g2, b2t],
      [out],
      (b, args) => {
        const q = b.matmul(args[0], args[1]).getResult(0);
        const k = b.matmul(args[0], args[2]).getResult(0);
        const v = b.matmul(args[0], args[3]).getResult(0);
        const kt = b.transpose(k, [1, 0]).getResult(0);
        const scores = b.matmul(q, kt).getResult(0);
        const sv = b.scalarConstant(1 / Math.sqrt(D_MODEL), ScalarType.F32).getResult(0);
        const sbc = b.broadcast(sv, [SEQ, SEQ], []).getResult(0);
        const scaled = b.mul(scores, sbc).getResult(0);
        const masked = b.add(scaled, args[5]).getResult(0);
        const attn = b.softmax(masked).getResult(0);
        const ctx = b.matmul(attn, v).getResult(0);
        const proj = b.matmul(ctx, args[4]).getResult(0);
        const res1 = b.add(args[0], proj).getResult(0);
        const norm1 = b.layernorm(res1, args[6], args[7]).getResult(0);

        const ff1 = b.matmul(norm1, args[8]).getResult(0);
        const act = b.gelu(ff1).getResult(0);
        const ff2 = b.matmul(act, args[9]).getResult(0);
        const res2 = b.add(norm1, ff2).getResult(0);
        const norm2 = b.layernorm(res2, args[10], args[11]).getResult(0);
        b.returnOp([norm2]);
      }
    );

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const p = encoderParams(500);
    const causalMask = new Float32Array(SEQ * SEQ);
    for (let i = 0; i < SEQ; i++)
      for (let j = 0; j < SEQ; j++)
        causalMask[i * SEQ + j] = j <= i ? 0 : -1e9;
    const [result] = run(r, 'decoder_block',
      [p.x, p.wq, p.wk, p.wv, p.wo, causalMask, p.g1, p.b1, p.wff1, p.wff2, p.g2, p.b2],
      [[SEQ, D_MODEL]]);

    expectMatches(result, ref.transformerEncoderBlock(p.x, { ...DIMS, ...p, mask: causalMask }), 'decoder_block');

    const unmasked = ref.transformerEncoderBlock(p.x, { ...DIMS, ...p });
    const differs = ref.expectClose(result, unmasked, 1e-3, 'decoder vs unmasked');
    expect(differs.ok, 'causal mask had no effect — masked output equals the unmasked reference').toBe(false);
  });

  it('compiles embedding + positional encoding', () => {
    const indices  = new TensorType([SEQ], ScalarType.I32);
    const tokEmb   = new TensorType([32, D_MODEL], ScalarType.F32);
    const posEmb   = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const out      = new TensorType([SEQ, D_MODEL], ScalarType.F32);

    const func = buildFunction('embed_pos', [indices, tokEmb, posEmb], [out], (b, args) => {
      const tok = b.embedding(args[1], args[0]).getResult(0);
      const combined = b.add(tok, args[2]).getResult(0);
      b.returnOp([combined]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const idxData = new Int32Array([1, 5, 3, 0]);
    const tokData = new Float32Array(32 * D_MODEL).fill(0).map((_, i) => (i % D_MODEL) * 0.1);
    const posData = new Float32Array(SEQ * D_MODEL).fill(0).map((_, i) => Math.sin(i * 0.01));
    const [result] = run(r, 'embed_pos', [idxData, tokData, posData], [[SEQ, D_MODEL]]);
    expect(result.every(v => isFinite(v))).toBe(true);
    for (let i = 0; i < D_MODEL; i++) {
      const tokVal = tokData[1 * D_MODEL + i];
      const posVal = posData[0 * D_MODEL + i];
      expect(result[i]).toBeCloseTo(tokVal + posVal, 4);
    }
  });

  it('compiles classification head: layernorm -> linear -> softmax', () => {
    const x     = new TensorType([SEQ, D_MODEL], ScalarType.F32);
    const gamma = new TensorType([D_MODEL], ScalarType.F32);
    const beta  = new TensorType([D_MODEL], ScalarType.F32);
    const wCls  = new TensorType([D_MODEL, 10], ScalarType.F32);
    const out   = new TensorType([SEQ, 10], ScalarType.F32);

    const func = buildFunction('cls_head', [x, gamma, beta, wCls], [out], (b, args) => {
      const normed = b.layernorm(args[0], args[1], args[2]).getResult(0);
      const logits = b.matmul(normed, args[3]).getResult(0);
      const probs = b.softmax(logits).getResult(0);
      b.returnOp([probs]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array(SEQ * D_MODEL).map((_, i) => Math.sin(i));
    const gData = new Float32Array(D_MODEL).fill(1);
    const bData = new Float32Array(D_MODEL).fill(0);
    const wData = new Float32Array(D_MODEL * 10).fill(0.1);
    const [result] = run(r, 'cls_head', [xData, gData, bData, wData], [[SEQ, 10]]);
    expect(result.every(v => isFinite(v) && v >= 0 && v <= 1)).toBe(true);
    for (let row = 0; row < SEQ; row++) {
      let sum = 0;
      for (let j = 0; j < 10; j++) sum += result[row * 10 + j];
      expect(sum).toBeCloseTo(1.0, 4);
    }
  });
});
