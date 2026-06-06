import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../../src/backend/target.js';

function compile(func, opts = {}) {
  return compileGraph(func, CPUTarget(), opts);
}

function countLoops(src) {
  return (src.match(/\bfor\s*\(/g) || []).length;
}

function countTempBuffers(src) {
  return (src.match(/new Float32Array/g) || []).length;
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
  it('compiles full CNN block pipeline', () => {
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

    const xData = new Float32Array(64).fill(1);
    const kData = new Float32Array(36).fill(1/9);
    const gData = new Float32Array([1, 1, 1, 1]);
    const bData = new Float32Array([0, 0, 0, 0]);
    const mData = new Float32Array([0, 0, 0, 0]);
    const vData = new Float32Array([1, 1, 1, 1]);
    const result = new Float32Array(36);
    r.run('cnn_block', xData, kData, gData, bData, mData, vData, result);
    expect(result.every(v => isFinite(v))).toBe(true);
  });
});

describe('Residual block: conv -> relu -> conv -> add(input) -> relu', () => {
  it('compiles residual skip connection', () => {
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

    const xData = new Float32Array(256).fill(0.5);
    const k1Data = new Float32Array(144).fill(0.1);
    const k2Data = new Float32Array(144).fill(0.1);
    const result = new Float32Array(64);
    r.run('resblock', xData, k1Data, k2Data, result);
    expect(result.every(v => isFinite(v) && v >= 0)).toBe(true);
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
  it('compiles feed-forward network with residual + layernorm', () => {
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

    const xData  = new Float32Array(8).fill(1);
    const w1Data = new Float32Array(32).fill(0.1);
    const b1Data = new Float32Array(16).fill(0);
    const w2Data = new Float32Array(32).fill(0.1);
    const b2Data = new Float32Array(8).fill(0);
    const gData  = new Float32Array([1, 1, 1, 1]);
    const bData  = new Float32Array([0, 0, 0, 0]);
    const result = new Float32Array(8);
    r.run('transformer_ffn', xData, w1Data, b1Data, w2Data, b2Data, gData, bData, result);
    expect(result.every(v => isFinite(v))).toBe(true);
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
  it('compiles Q projection + head reshape', () => {
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

    const xData = new Float32Array(32).fill(0).map((_, i) => i / 32);
    const wData = new Float32Array(64);
    for (let i = 0; i < 8; i++) wData[i * 8 + i] = 1;
    const result = new Float32Array(32);
    r.run('mha_proj', xData, wData, result);
    expect(result.every(v => isFinite(v))).toBe(true);
  });
});

describe('Bottleneck: 1x1 conv -> relu -> 3x3 conv -> relu -> 1x1 conv', () => {
  it('compiles 3-conv bottleneck block', () => {
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

    const xData  = new Float32Array(512).fill(0.5);
    const k1Data = new Float32Array(32).fill(0.1);
    const k2Data = new Float32Array(144).fill(0.05);
    const k3Data = new Float32Array(32).fill(0.1);
    const result = new Float32Array(288);
    r.run('bottleneck', xData, k1Data, k2Data, k3Data, result);
    expect(result.every(v => isFinite(v))).toBe(true);
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
  it('compiles gated activation pattern', () => {
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

    const xData    = new Float32Array(8).fill(1);
    const wGData   = new Float32Array(16).fill(0.5);
    const wUData   = new Float32Array(16).fill(0.5);
    const gammaD   = new Float32Array([1, 1, 1, 1]);
    const betaD    = new Float32Array([0, 0, 0, 0]);
    const result   = new Float32Array(8);
    r.run('swiglu', xData, wGData, wUData, gammaD, betaD, result);
    expect(result.every(v => isFinite(v))).toBe(true);
  });
});

describe('Depthwise separable: depthwise conv -> pointwise conv -> relu', () => {
  it('compiles depthwise separable convolution', () => {
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

    const xData  = new Float32Array(256).fill(1);
    const dwData = new Float32Array(36).fill(1/9);
    const pwData = new Float32Array(32).fill(0.1);
    const result = new Float32Array(288);
    r.run('dw_sep', xData, dwData, pwData, result);
    expect(result.every(v => isFinite(v) && v >= 0)).toBe(true);
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
  it('compiles parallel conv paths with different strides', () => {
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

    const xData  = new Float32Array(64).fill(1);
    const k1Data = new Float32Array(9).fill(1);
    const k2Data = new Float32Array(9).fill(1);
    const res1   = new Float32Array(36);
    const res2   = new Float32Array(9);
    r.run('multi_scale', xData, k1Data, k2Data, res1, res2);
    expect(res1.every(v => isFinite(v) && v >= 0)).toBe(true);
    expect(res2.every(v => isFinite(v) && v >= 0)).toBe(true);
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
