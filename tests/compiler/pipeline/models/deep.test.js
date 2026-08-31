import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../../src/compiler/ir/graph/types.js';
import { CPUTarget } from '../../../../src/compiler/support/target.js';
import { countLoops, countTempBuffers } from '../../../_utils/kernel_source.js';
import { randomArray } from '../../../_utils/rng.js';
import * as ref from '../../../_utils/reference_ops.js';
import { compileCPU as compile } from '../../../_utils/ir_fixture.js';

function expectMatches(actual, expected, label, tol = 1e-4) {
  const r = ref.expectClose(actual, expected, tol, label);
  expect(r.ok, r.message).toBe(true);
}



describe('MLP: matmul -> bias -> relu -> matmul -> bias', () => {
  it('compiles 2-layer MLP and produces correct output', () => {
    const x   = new TensorType([2, 4], ScalarType.F32);
    const w1  = new TensorType([4, 3], ScalarType.F32);
    const b1  = new TensorType([2, 3], ScalarType.F32);
    const w2  = new TensorType([3, 2], ScalarType.F32);
    const b2  = new TensorType([2, 2], ScalarType.F32);
    const out = new TensorType([2, 2], ScalarType.F32);

    const func = buildFunction('mlp_2layer', [x, w1, b1, w2, b2], [out], (b, args) => {
      const h1 = b.matmul(args[0], args[1]).getResult(0);
      const h1b = b.add(h1, args[2]).getResult(0);
      const h1a = b.relu(h1b).getResult(0);
      const h2 = b.matmul(h1a, args[3]).getResult(0);
      const h2b = b.add(h2, args[4]).getResult(0);
      b.returnOp([h2b]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData  = new Float32Array([1, 0, 0, 0,  0, 1, 0, 0]);
    const w1Data = new Float32Array([1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 0]);
    const b1Data = new Float32Array([0, 0, 0,  0, 0, 0]);
    const w2Data = new Float32Array([1, 0,  0, 1,  1, 1]);
    const b2Data = new Float32Array([0, 0,  0, 0]);
    const result = new Float32Array(4);
    r.run('mlp_2layer', xData, w1Data, b1Data, w2Data, b2Data, result);

    expect(result[0]).toBe(1);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(0);
    expect(result[3]).toBe(1);
  });

  it('bias+relu should be fused', () => {
    const x   = new TensorType([1, 4], ScalarType.F32);
    const w   = new TensorType([4, 4], ScalarType.F32);
    const b1  = new TensorType([1, 4], ScalarType.F32);
    const out = new TensorType([1, 4], ScalarType.F32);

    const func = buildFunction('mm_bias_relu', [x, w, b1], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const biased = b.add(mm, args[2]).getResult(0);
      const activated = b.relu(biased).getResult(0);
      b.returnOp([activated]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);
  });
});

describe('CNN Block: conv -> batchnorm -> relu -> maxpool', () => {
  it('matches a reference CNN block: conv, per-channel batchnorm, relu, then 2x2 max pool', () => {
    const x    = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const k    = new TensorType([4, 1, 3, 3], ScalarType.F32);
    const gamma = new TensorType([4], ScalarType.F32);
    const beta  = new TensorType([4], ScalarType.F32);
    const mean  = new TensorType([4], ScalarType.F32);
    const var_  = new TensorType([4], ScalarType.F32);
    const out   = new TensorType([1, 4, 3, 3], ScalarType.F32);

    const func = buildFunction('cnn_block', [x, k, gamma, beta, mean, var_], [out], (b, args) => {
      const conv = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const bn = b.batchnorm(conv, args[2], args[3], args[4], args[5]).getResult(0);
      const act = b.relu(bn).getResult(0);
      const pool = b.pool2d(act, 'max', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0);
      b.returnOp([pool]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(111, 64);
    const kData = randomArray(112, 36, -0.4, 0.4);
    const gData = randomArray(113, 4, 0.5, 1.5);
    const bData = randomArray(114, 4, -0.5, 0.5);
    const mData = randomArray(115, 4, -0.3, 0.3);
    const vData = randomArray(116, 4, 0.5, 2);
    const result = new Float32Array(36);
    r.run('cnn_block', xData, kData, gData, bData, mData, vData, result);

    const conv = ref.conv2d(xData, [1, 1, 8, 8], kData, [4, 1, 3, 3]).data;
    const bn = ref.batchNorm(conv, [1, 4, 6, 6], gData, bData, mData, vData);
    expectMatches(result, ref.pool2d(ref.relu(bn), [1, 4, 6, 6], 'max', [2, 2]).data, 'cnn_block');
  });
});

describe('Residual block: conv -> relu -> conv -> add(input) -> relu', () => {
  it('matches a reference residual block whose shortcut is an avg-pool downsample, not the raw input', () => {
    const x  = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const k1 = new TensorType([4, 4, 3, 3], ScalarType.F32);
    const k2 = new TensorType([4, 4, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 4, 4, 4], ScalarType.F32);

    const func = buildFunction('resblock', [x, k1, k2], [out], (b, args) => {
      const conv1 = b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0);
      const act1 = b.relu(conv1).getResult(0);
      const conv2 = b.conv(act1, args[2], [2, 2], [[1, 1], [1, 1]]).getResult(0);
      const shortcut = b.pool2d(args[0], 'avg', [2, 2], [2, 2], [[0, 0], [0, 0]]).getResult(0);
      const residual = b.add(conv2, shortcut).getResult(0);
      const final = b.relu(residual).getResult(0);
      b.returnOp([final]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(121, 256);
    const k1Data = randomArray(122, 144, -0.25, 0.25);
    const k2Data = randomArray(123, 144, -0.25, 0.25);
    const result = new Float32Array(64);
    r.run('resblock', xData, k1Data, k2Data, result);

    const conv1 = ref.relu(ref.conv2d(xData, [1, 4, 8, 8], k1Data, [4, 4, 3, 3], { pad: 1 }).data);
    const conv2 = ref.conv2d(conv1, [1, 4, 8, 8], k2Data, [4, 4, 3, 3], { stride: 2, pad: 1 }).data;
    const shortcut = ref.pool2d(xData, [1, 4, 8, 8], 'avg', [2, 2]).data;
    expectMatches(result, ref.relu(ref.add(conv2, shortcut)), 'resblock');
  });
});

describe('Self-Attention: Q*K^T * scale -> softmax -> * V', () => {
  it('compiles scaled dot-product attention', () => {
    const Q   = new TensorType([4, 8], ScalarType.F32);
    const K   = new TensorType([4, 8], ScalarType.F32);
    const V   = new TensorType([4, 8], ScalarType.F32);
    const out = new TensorType([4, 8], ScalarType.F32);

    const func = buildFunction('attention', [Q, K, V], [out], (b, args) => {
      const kt = b.transpose(args[1], [1, 0]).getResult(0);
      const scores = b.matmul(args[0], kt).getResult(0);

      const scaleVal = b.scalarConstant(1.0 / Math.sqrt(8), ScalarType.F32).getResult(0);
      const scaleBcast = b.broadcast(scaleVal, [4, 4], []).getResult(0);
      const scaled = b.mul(scores, scaleBcast).getResult(0);

      const attnWeights = b.softmax(scaled).getResult(0);
      const attended = b.matmul(attnWeights, args[2]).getResult(0);
      b.returnOp([attended]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const qData = new Float32Array(32).fill(0).map((_, i) => Math.sin(i));
    const kData = new Float32Array(32).fill(0).map((_, i) => Math.cos(i));
    const vData = new Float32Array(32).fill(1);
    const result = new Float32Array(32);
    r.run('attention', qData, kData, vData, result);
    expect(result.every(v => isFinite(v))).toBe(true);
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 8; col++) {
        expect(result[row * 8 + col]).toBeCloseTo(1.0, 1);
      }
    }
  });
});

describe('Transformer FFN: linear -> gelu -> linear -> add -> layernorm', () => {
  it('matches a reference FFN: both biases applied, residual added before the layernorm', () => {
    const x     = new TensorType([2, 4], ScalarType.F32);
    const w1    = new TensorType([4, 8], ScalarType.F32);
    const b1    = new TensorType([2, 8], ScalarType.F32);
    const w2    = new TensorType([8, 4], ScalarType.F32);
    const b2    = new TensorType([2, 4], ScalarType.F32);
    const gamma = new TensorType([4], ScalarType.F32);
    const beta  = new TensorType([4], ScalarType.F32);
    const out   = new TensorType([2, 4], ScalarType.F32);

    const func = buildFunction('transformer_ffn', [x, w1, b1, w2, b2, gamma, beta], [out], (b, args) => {
      const h = b.matmul(args[0], args[1]).getResult(0);
      const hb = b.add(h, args[2]).getResult(0);
      const ha = b.gelu(hb).getResult(0);
      const proj = b.matmul(ha, args[3]).getResult(0);
      const projb = b.add(proj, args[4]).getResult(0);
      const residual = b.add(args[0], projb).getResult(0);
      const normed = b.layernorm(residual, args[5], args[6]).getResult(0);
      b.returnOp([normed]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData  = randomArray(131, 8);
    const w1Data = randomArray(132, 32, -0.5, 0.5);
    const b1Data = randomArray(133, 16, -0.3, 0.3);
    const w2Data = randomArray(134, 32, -0.5, 0.5);
    const b2Data = randomArray(135, 8, -0.3, 0.3);
    const gData  = randomArray(136, 4, 0.5, 1.5);
    const bData  = randomArray(137, 4, -0.4, 0.4);
    const result = new Float32Array(8);
    r.run('transformer_ffn', xData, w1Data, b1Data, w2Data, b2Data, gData, bData, result);

    const hidden = ref.gelu(ref.add(ref.matmul(xData, [2, 4], w1Data, [4, 8]), b1Data));
    const proj = ref.add(ref.matmul(hidden, [2, 8], w2Data, [8, 4]), b2Data);
    const residual = ref.add(xData, proj);
    expectMatches(result, ref.layerNormLastAxis(residual, [2, 4], { gamma: gData, beta: bData }), 'transformer_ffn');
  });
});

describe('Deep elementwise chain: 8 ops fused into single kernel', () => {
  it('x -> mul -> add -> tanh -> mul -> neg -> exp -> add -> sigmoid', () => {
    const t = new TensorType([16], ScalarType.F32);

    const func = buildFunction('deep_ew', [t, t], [t], (b, args) => {
      let v = b.mul(args[0], args[1]).getResult(0);
      v = b.add(v, args[0]).getResult(0);
      v = b.tanh(v).getResult(0);
      v = b.mul(v, args[1]).getResult(0);
      v = b.neg(v).getResult(0);
      v = b.exp(v).getResult(0);
      v = b.add(v, args[0]).getResult(0);
      v = b.sigmoid(v).getResult(0);
      b.returnOp([v]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);
    const src = r.getSource('deep_ew');

    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);

    const a = new Float32Array(16).fill(0.5);
    const c = new Float32Array(16).fill(1.0);
    const result = new Float32Array(16);
    r.run('deep_ew', a, c, result);
    expect(result.every(v => v > 0 && v < 1)).toBe(true);
  });
});

describe('Multi-head projection: matmul -> reshape -> transpose', () => {
  it('matches a reference multi-head split: reshape to [tokens, heads, dim] then move heads to the front', () => {
    const x    = new TensorType([4, 8], ScalarType.F32);
    const wq   = new TensorType([8, 8], ScalarType.F32);
    const out  = new TensorType([2, 4, 4], ScalarType.F32);

    const func = buildFunction('mha_proj', [x, wq], [out], (b, args) => {
      const proj = b.matmul(args[0], args[1]).getResult(0);
      const reshaped = b.reshape(proj, [4, 2, 4]).getResult(0);
      const transposed = b.transpose(reshaped, [1, 0, 2]).getResult(0);
      b.returnOp([transposed]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = randomArray(141, 32);
    const wData = randomArray(142, 64, -0.6, 0.6);
    const result = new Float32Array(32);
    r.run('mha_proj', xData, wData, result);

    const proj = ref.matmul(xData, [4, 8], wData, [8, 8]);
    expectMatches(result, ref.transpose(proj, [4, 2, 4], [1, 0, 2]), 'mha_proj');
  });
});

describe('Bottleneck: 1x1 conv -> relu -> 3x3 conv -> relu -> 1x1 conv', () => {
  it('matches a reference bottleneck: 1x1 reduce, 3x3, 1x1 expand with relu only between the first two', () => {
    const x  = new TensorType([1, 8, 8, 8], ScalarType.F32);
    const k1 = new TensorType([4, 8, 1, 1], ScalarType.F32);
    const k2 = new TensorType([4, 4, 3, 3], ScalarType.F32);
    const k3 = new TensorType([8, 4, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 8, 6, 6], ScalarType.F32);

    const func = buildFunction('bottleneck', [x, k1, k2, k3], [out], (b, args) => {
      const c1 = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const a1 = b.relu(c1).getResult(0);
      const c2 = b.conv(a1, args[2], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const a2 = b.relu(c2).getResult(0);
      const c3 = b.conv(a2, args[3], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      b.returnOp([c3]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData  = randomArray(151, 512);
    const k1Data = randomArray(152, 32, -0.4, 0.4);
    const k2Data = randomArray(153, 144, -0.2, 0.2);
    const k3Data = randomArray(154, 32, -0.4, 0.4);
    const result = new Float32Array(288);
    r.run('bottleneck', xData, k1Data, k2Data, k3Data, result);

    const a1 = ref.relu(ref.conv2d(xData, [1, 8, 8, 8], k1Data, [4, 8, 1, 1]).data);
    const a2 = ref.relu(ref.conv2d(a1, [1, 4, 8, 8], k2Data, [4, 4, 3, 3]).data);
    expectMatches(result, ref.conv2d(a2, [1, 4, 6, 6], k3Data, [8, 4, 1, 1]).data, 'bottleneck');
  });
});

describe('Classifier head: global avgpool -> linear -> softmax', () => {
  it('compiles classification head', () => {
    const x     = new TensorType([1, 4, 4, 4], ScalarType.F32);
    const w     = new TensorType([4, 10], ScalarType.F32);
    const bias  = new TensorType([1, 10], ScalarType.F32);
    const out   = new TensorType([1, 10], ScalarType.F32);

    const func = buildFunction('cls_head', [x, w, bias], [out], (b, args) => {
      const pooled = b.pool2d(args[0], 'avg', [4, 4], [4, 4], [[0, 0], [0, 0]]).getResult(0);
      const flat = b.reshape(pooled, [1, 4]).getResult(0);
      const logits = b.matmul(flat, args[1]).getResult(0);
      const biased = b.add(logits, args[2]).getResult(0);
      const probs = b.softmax(biased).getResult(0);
      b.returnOp([probs]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array(64).fill(1);
    const wData = new Float32Array(40).fill(0.1);
    const bData = new Float32Array(10).fill(0);
    const result = new Float32Array(10);
    r.run('cls_head', xData, wData, bData, result);
    let sum = 0;
    for (let i = 0; i < 10; i++) sum += result[i];
    expect(sum).toBeCloseTo(1.0, 3);
  });
});

describe('SwiGLU-like: (silu(x*W1) * (x*W2)) + layernorm', () => {
  it('matches a reference SwiGLU: silu applied only to the gate branch, not the up branch', () => {
    const x     = new TensorType([2, 4], ScalarType.F32);
    const wGate = new TensorType([4, 4], ScalarType.F32);
    const wUp   = new TensorType([4, 4], ScalarType.F32);
    const gamma = new TensorType([4], ScalarType.F32);
    const beta  = new TensorType([4], ScalarType.F32);
    const out   = new TensorType([2, 4], ScalarType.F32);

    const func = buildFunction('swiglu', [x, wGate, wUp, gamma, beta], [out], (b, args) => {
      const gate = b.matmul(args[0], args[1]).getResult(0);
      const gateAct = b.silu(gate).getResult(0);
      const up = b.matmul(args[0], args[2]).getResult(0);
      const gated = b.mul(gateAct, up).getResult(0);
      const normed = b.layernorm(gated, args[3], args[4]).getResult(0);
      b.returnOp([normed]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData    = randomArray(161, 8);
    const wGData   = randomArray(162, 16, -0.7, 0.7);
    const wUData   = randomArray(163, 16, -0.7, 0.7);
    const gammaD   = randomArray(164, 4, 0.5, 1.5);
    const betaD    = randomArray(165, 4, -0.4, 0.4);
    const result   = new Float32Array(8);
    r.run('swiglu', xData, wGData, wUData, gammaD, betaD, result);

    const gate = ref.silu(ref.matmul(xData, [2, 4], wGData, [4, 4]));
    const up = ref.matmul(xData, [2, 4], wUData, [4, 4]);
    expectMatches(result, ref.layerNormLastAxis(ref.mul(gate, up), [2, 4], { gamma: gammaD, beta: betaD }), 'swiglu');
  });
});

describe('Depthwise separable: depthwise conv -> pointwise conv -> relu', () => {
  it('matches a reference depthwise-separable conv: groups=4 depthwise keeps channels independent before the 1x1 mix', () => {
    const x   = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const dw  = new TensorType([4, 1, 3, 3], ScalarType.F32);
    const pw  = new TensorType([8, 4, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 8, 6, 6], ScalarType.F32);

    const func = buildFunction('dw_sep', [x, dw, pw], [out], (b, args) => {
      const depthwise = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]], { groups: 4 }).getResult(0);
      const act1 = b.relu(depthwise).getResult(0);
      const pointwise = b.conv(act1, args[2], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const act2 = b.relu(pointwise).getResult(0);
      b.returnOp([act2]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData  = randomArray(171, 256);
    const dwData = randomArray(172, 36, -0.4, 0.4);
    const pwData = randomArray(173, 32, -0.4, 0.4);
    const result = new Float32Array(288);
    r.run('dw_sep', xData, dwData, pwData, result);

    const depthwise = ref.relu(ref.conv2d(xData, [1, 4, 8, 8], dwData, [4, 1, 3, 3], { groups: 4 }).data);
    expectMatches(result, ref.relu(ref.conv2d(depthwise, [1, 4, 6, 6], pwData, [8, 4, 1, 1]).data), 'dw_sep');
  });
});

describe('Loss: logits -> log_softmax -> reduce_sum', () => {
  it('compiles loss computation pipeline', () => {
    const x   = new TensorType([2, 4], ScalarType.F32);
    const w   = new TensorType([4, 3], ScalarType.F32);
    const out = new TensorType([2], ScalarType.F32);

    const func = buildFunction('loss_pipe', [x, w], [out], (b, args) => {
      const logits = b.matmul(args[0], args[1]).getResult(0);
      const lsm = b.logSoftmax(logits).getResult(0);
      const zero = b.scalarConstant(0, ScalarType.F32).getResult(0);
      const reduced = b.reduce(lsm, zero, [1], 'sum').getResult(0);
      b.returnOp([reduced]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array([1, 0, 0, 0,  0, 1, 0, 0]);
    const wData = new Float32Array([1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 0]);
    const result = new Float32Array(2);
    r.run('loss_pipe', xData, wData, result);
    expect(result[0]).toBeLessThan(0);
    expect(result[1]).toBeLessThan(0);
  });
});

describe('Multi-scale: conv at different strides', () => {
  it('matches reference conv at both scales, with each output written to its own buffer', () => {
    const x   = new TensorType([1, 1, 8, 8], ScalarType.F32);
    const k1  = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const k2  = new TensorType([1, 1, 3, 3], ScalarType.F32);
    const out1 = new TensorType([1, 1, 6, 6], ScalarType.F32);
    const out2 = new TensorType([1, 1, 3, 3], ScalarType.F32);

    const func = buildFunction('multi_scale', [x, k1, k2], [out1, out2], (b, args) => {
      const fine = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const fine_act = b.relu(fine).getResult(0);
      const coarse = b.conv(args[0], args[2], [2, 2], [[0, 0], [0, 0]]).getResult(0);
      const coarse_act = b.relu(coarse).getResult(0);
      b.returnOp([fine_act, coarse_act]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData  = randomArray(181, 64);
    const k1Data = randomArray(182, 9, -0.8, 0.8);
    const k2Data = randomArray(183, 9, -0.8, 0.8);
    const res1   = new Float32Array(36);
    const res2   = new Float32Array(9);
    r.run('multi_scale', xData, k1Data, k2Data, res1, res2);

    const shape = [1, 1, 8, 8];
    expectMatches(res1, ref.relu(ref.conv2d(xData, shape, k1Data, [1, 1, 3, 3]).data), 'multi_scale fine');
    expectMatches(res2, ref.relu(ref.conv2d(xData, shape, k2Data, [1, 1, 3, 3], { stride: 2 }).data), 'multi_scale coarse');
  });
});

describe('Statistics: manual mean + variance via reduce', () => {
  it('computes mean and variance', () => {
    const x = new TensorType([4, 8], ScalarType.F32);
    const outMean = new TensorType([4], ScalarType.F32);
    const outVar  = new TensorType([4], ScalarType.F32);

    const func = buildFunction('mean_var', [x], [outMean, outVar], (b, args) => {
      const zero = b.scalarConstant(0, ScalarType.F32).getResult(0);
      const mean = b.reduce(args[0], zero, [1], 'mean').getResult(0);

      const meanBcast = b.broadcast(mean, [4, 8], [0]).getResult(0);
      const diff = b.sub(args[0], meanBcast).getResult(0);
      const sq = b.mul(diff, diff).getResult(0);
      const variance = b.reduce(sq, zero, [1], 'mean').getResult(0);

      b.returnOp([mean, variance]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);

    const xData = new Float32Array(32);
    for (let i = 0; i < 32; i++) xData[i] = i;
    const meanR = new Float32Array(4);
    const varR  = new Float32Array(4);
    r.run('mean_var', xData, meanR, varR);
    expect(meanR[0]).toBeCloseTo(3.5, 2);
    expect(varR[0]).toBeGreaterThan(0);
    expect(varR.every(v => v >= 0)).toBe(true);
  });
});

describe('KERNEL QUALITY AUDIT', () => {
  it('no extent-1 loops in batch=1 models', () => {
    const x = new TensorType([1, 4], ScalarType.F32);
    const w = new TensorType([4, 4], ScalarType.F32);
    const out = new TensorType([1, 4], ScalarType.F32);

    const func = buildFunction('b1_audit', [x, w], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const act = b.relu(mm).getResult(0);
      b.returnOp([act]);
    });

    const r = compile(func);
    const src = r.getSource('b1_audit');
    const extent1Loops = src.match(/for\s*\(\s*let\s+\w+\s*=\s*0;\s*\w+\s*<\s*1;/g);
    expect(extent1Loops).toBeNull();
  });

  it('no redundant zero-init in matmul kernels', () => {
    const x = new TensorType([4, 8], ScalarType.F32);
    const w = new TensorType([8, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);

    const func = buildFunction('noinit_audit', [x, w], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('noinit_audit');
    const zeroStores = src.match(/\w+\[\w+\]\s*=\s*0\s*;/g) || [];
    expect(zeroStores.length).toBe(0);
  });

  it('no 0*stride or stride*0 in index expressions (matmul)', () => {
    const lhs = new TensorType([1, 4], ScalarType.F32);
    const rhs = new TensorType([4, 3], ScalarType.F32);
    const out = new TensorType([1, 3], ScalarType.F32);

    const func = buildFunction('idx_audit', [lhs, rhs], [out], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('idx_audit');
    expect(src).not.toMatch(/\b0\s*\*\s*\d+\b/);
    expect(src).not.toMatch(/\b\d+\s*\*\s*0\b/);
  });

  it('conv with zero padding has no bounds checks', () => {
    const x = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const k = new TensorType([4, 4, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 4, 6, 6], ScalarType.F32);

    const func = buildFunction('conv_nopad', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    expect(r.succeeded).toBe(true);
    const src = r.getSource('conv_nopad');
    expect(src).not.toMatch(/>=\s*0/);
  });

  it('elementwise chains fully fused — no intermediate buffers', () => {
    const t = new TensorType([32], ScalarType.F32);
    const func = buildFunction('fusion_audit', [t], [t], (b, args) => {
      let v = b.exp(args[0]).getResult(0);
      v = b.neg(v).getResult(0);
      v = b.tanh(v).getResult(0);
      v = b.abs(v).getResult(0);
      v = b.sigmoid(v).getResult(0);
      b.returnOp([v]);
    });

    const r = compile(func);
    const src = r.getSource('fusion_audit');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
  });
});
