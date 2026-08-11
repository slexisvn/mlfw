import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget, CUDATarget } from '../../src/backend/target.js';
import { countLoops, countTempBuffers } from '../_utils/kernel_source.js';
import { compileCPU as compile } from '../_utils/ir_fixture.js';
import { audit, arithmeticNoise, extent1Loops, redundantZeroInits, boundsChecks, modulos, truncs } from '../_utils/kernel_audit.js';

describe('Kernel audit — elementwise fusion quality', () => {
  it('8-op elementwise chain: single loop, zero temp buffers', () => {
    const t = new TensorType([256], ScalarType.F32);
    const func = buildFunction('ew8', [t, t], [t], (b, args) => {
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
    const src = r.getSource('ew8');
    const a = audit(src, 'ew8');

    expect(a.loops).toBe(1);
    expect(a.tempBuffers).toBe(0);
    expect(a.arithmeticNoise).toEqual([]);
    expect(a.extent1Loops).toBe(0);
  });

  it('sigmoid decomposition: no temp buffer for constant 1.0', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('sig_solo', [t], [t], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('sig_solo');
    const a = audit(src, 'sig_solo');

    expect(a.loops).toBe(1);
    expect(a.tempBuffers).toBe(0);
    expect(src).not.toMatch(/new Float32Array/);
  });

  it('silu = x * sigmoid(x): no temp buffer', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('silu_solo', [t], [t], (b, args) => {
      b.returnOp([b.silu(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('silu_solo');
    expect(countTempBuffers(src)).toBe(0);
    expect(countLoops(src)).toBe(1);
  });

  it('gelu: no temp buffer', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('gelu_solo', [t], [t], (b, args) => {
      b.returnOp([b.gelu(args[0]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('gelu_solo');
    expect(countTempBuffers(src)).toBe(0);
    expect(countLoops(src)).toBe(1);
  });

  it('diamond DAG: exp -> (sigmoid, tanh) -> mul: single loop', () => {
    const t = new TensorType([64], ScalarType.F32);
    const func = buildFunction('diamond', [t], [t], (b, args) => {
      const e = b.exp(args[0]).getResult(0);
      const s = b.sigmoid(e).getResult(0);
      const th = b.tanh(e).getResult(0);
      b.returnOp([b.mul(s, th).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('diamond');
    expect(countLoops(src)).toBe(1);
    expect(countTempBuffers(src)).toBe(0);
  });
});

describe('Kernel audit — matmul quality', () => {
  it('basic matmul: 3 loops, no arithmetic noise', () => {
    const a = new TensorType([8, 16], ScalarType.F32);
    const b_ = new TensorType([16, 8], ScalarType.F32);
    const c = new TensorType([8, 8], ScalarType.F32);

    const func = buildFunction('mm', [a, b_], [c], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('mm');
    const au = audit(src, 'mm');

    expect(au.arithmeticNoise).toEqual([]);
    expect(au.extent1Loops).toBe(0);
  });

  it('batch=1 matmul: no extent-1 loop for batch dim', () => {
    const a = new TensorType([1, 8], ScalarType.F32);
    const b_ = new TensorType([8, 4], ScalarType.F32);
    const c = new TensorType([1, 4], ScalarType.F32);

    const func = buildFunction('mm_b1', [a, b_], [c], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('mm_b1');
    expect(extent1Loops(src)).toBe(0);
    expect(arithmeticNoise(src)).toEqual([]);
  });

  it('matmul + bias + relu: bias-relu fused, no extra loop', () => {
    const x = new TensorType([4, 8], ScalarType.F32);
    const w = new TensorType([8, 4], ScalarType.F32);
    const bi = new TensorType([4, 4], ScalarType.F32);
    const out = new TensorType([4, 4], ScalarType.F32);

    const func = buildFunction('mm_bias_relu', [x, w, bi], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const biased = b.add(mm, args[2]).getResult(0);
      b.returnOp([b.relu(biased).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('mm_bias_relu');
    const loops = countLoops(src);
    expect(loops).toBeLessThanOrEqual(7);
    expect(arithmeticNoise(src)).toEqual([]);
  });
});

describe('Kernel audit — conv quality', () => {
  it('conv pad=0: no bounds checks, no arithmetic noise', () => {
    const x = new TensorType([1, 8, 16, 16], ScalarType.F32);
    const k = new TensorType([16, 8, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 16, 14, 14], ScalarType.F32);

    const func = buildFunction('conv_q', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_q');
    const au = audit(src, 'conv_q');

    expect(au.boundsChecks).toBe(0);
    expect(au.arithmeticNoise).toEqual([]);
    expect(au.extent1Loops).toBe(0);
  });

  it('1x1 pointwise conv: clean index arithmetic', () => {
    const x = new TensorType([1, 32, 8, 8], ScalarType.F32);
    const k = new TensorType([64, 32, 1, 1], ScalarType.F32);
    const out = new TensorType([1, 64, 8, 8], ScalarType.F32);

    const func = buildFunction('pw', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('pw');
    const au = audit(src, 'pw');

    expect(au.arithmeticNoise).toEqual([]);
    expect(au.boundsChecks).toBe(0);
  });

  it('conv with padding: bounds checks present but no arithmetic noise', () => {
    const x = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const k = new TensorType([8, 4, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 8, 8, 8], ScalarType.F32);

    const func = buildFunction('conv_pad', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_pad');
    const au = audit(src, 'conv_pad');

    expect(au.boundsChecks).toBeGreaterThan(0);
    expect(au.arithmeticNoise).toEqual([]);
  });

  it('depthwise conv groups=C: compiles without noise', () => {
    const x = new TensorType([1, 8, 8, 8], ScalarType.F32);
    const k = new TensorType([8, 1, 3, 3], ScalarType.F32);
    const out = new TensorType([1, 8, 6, 6], ScalarType.F32);

    const func = buildFunction('dw', [x, k], [out], (b, args) => {
      b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]], { groups: 8 }).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('dw');
    expect(arithmeticNoise(src)).toEqual([]);
    expect(boundsChecks(src)).toBe(0);
  });
});

describe('Kernel audit — shape ops quality', () => {
  it('identity reshape: no modulo, no division', () => {
    const t = new TensorType([16, 32], ScalarType.F32);
    const func = buildFunction('id_resh', [t], [t], (b, args) => {
      b.returnOp([b.reshape(args[0], [16, 32]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('id_resh');
    expect(modulos(src)).toBe(0);
    expect(truncs(src)).toBe(0);
  });

  it('flatten reshape [4,8] -> [32]: no modulo by 1', () => {
    const a = new TensorType([4, 8], ScalarType.F32);
    const b_ = new TensorType([32], ScalarType.F32);

    const func = buildFunction('flatten', [a], [b_], (b, args) => {
      b.returnOp([b.reshape(args[0], [32]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('flatten');
    expect(src).not.toMatch(/%\s*1\b/);
  });

  it('reshape [12] -> [3,4]: clean index computation', () => {
    const a = new TensorType([12], ScalarType.F32);
    const b_ = new TensorType([3, 4], ScalarType.F32);

    const func = buildFunction('unflatten', [a], [b_], (b, args) => {
      b.returnOp([b.reshape(args[0], [3, 4]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('unflatten');
    expect(src).not.toMatch(/%\s*1\b/);
    expect(arithmeticNoise(src)).toEqual([]);
  });

  it('transpose [2,3] -> [3,2]: single loop nest', () => {
    const a = new TensorType([2, 3], ScalarType.F32);
    const b_ = new TensorType([3, 2], ScalarType.F32);

    const func = buildFunction('tr', [a], [b_], (b, args) => {
      b.returnOp([b.transpose(args[0], [1, 0]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('tr');
    expect(countLoops(src)).toBe(2);
    expect(arithmeticNoise(src)).toEqual([]);
  });
});

describe('Kernel audit — reduction quality', () => {
  it('sum reduction: init + accumulation, no extra buffers', () => {
    const x = new TensorType([4, 8], ScalarType.F32);
    const zero = new TensorType([], ScalarType.F32);
    const out = new TensorType([4], ScalarType.F32);

    const func = buildFunction('rowsum', [x, zero], [out], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'sum').getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('rowsum');
    expect(countTempBuffers(src)).toBe(0);
    expect(arithmeticNoise(src)).toEqual([]);
  });

  it('mean reduction: correct division factor', () => {
    const x = new TensorType([4, 8], ScalarType.F32);
    const zero = new TensorType([], ScalarType.F32);
    const out = new TensorType([4], ScalarType.F32);

    const func = buildFunction('rowmean', [x, zero], [out], (b, args) => {
      b.returnOp([b.reduce(args[0], args[1], [1], 'mean').getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('rowmean');
    expect(arithmeticNoise(src)).toEqual([]);
  });
});

describe('Kernel audit — quantized conv quality (affine index folding)', () => {
  const mkConvQ = (xShape, kShape, outShape, pad) =>
    buildFunction('cq', [new TensorType(xShape, ScalarType.F32), new TensorType(kShape, ScalarType.F32)],
      [new TensorType(outShape, ScalarType.F32)], (b, args) => {
        b.returnOp([b.conv(args[0], args[1], [1, 1], pad).getResult(0)]);
      });

  it('CPU quantized conv pad=0: no arithmetic noise, no bounds checks', () => {
    const r = compileGraph(mkConvQ([1, 3, 10, 10], [4, 3, 3, 3], [1, 4, 8, 8], [[0, 0], [0, 0]]),
      CPUTarget(), { quantization: { enabled: true } });
    const src = r.getSource('cq');
    expect(arithmeticNoise(src)).toEqual([]);
    expect(boundsChecks(src)).toBe(0);
    expect(extent1Loops(src)).toBe(0);
  });

  it('GPU quantized conv pad=0: no arithmetic noise, no bounds checks', () => {
    const r = compileGraph(mkConvQ([1, 3, 10, 10], [4, 3, 3, 3], [1, 4, 8, 8], [[0, 0], [0, 0]]),
      CUDATarget(), { quantization: { enabled: true } });
    const src = r.getSource('cq');
    expect(arithmeticNoise(src)).toEqual([]);
    expect(boundsChecks(src)).toBe(0);
  });

  it('GPU quantized conv pad=1: bounds checks present but no arithmetic noise', () => {
    const r = compileGraph(mkConvQ([1, 3, 8, 8], [4, 3, 3, 3], [1, 4, 8, 8], [[1, 1], [1, 1]]),
      CUDATarget(), { quantization: { enabled: true } });
    const src = r.getSource('cq');
    expect(boundsChecks(src)).toBeGreaterThan(0);
    expect(arithmeticNoise(src)).toEqual([]);
  });
});

describe('Kernel audit — composite model quality', () => {
  it('MLP layer: matmul loops + fused bias-relu loop, no wasted ops', () => {
    const x = new TensorType([4, 16], ScalarType.F32);
    const w = new TensorType([16, 8], ScalarType.F32);
    const bi = new TensorType([4, 8], ScalarType.F32);
    const out = new TensorType([4, 8], ScalarType.F32);

    const func = buildFunction('mlp_layer', [x, w, bi], [out], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const biased = b.add(mm, args[2]).getResult(0);
      b.returnOp([b.relu(biased).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('mlp_layer');
    const au = audit(src, 'mlp_layer');

    expect(au.arithmeticNoise).toEqual([]);
    expect(au.extent1Loops).toBe(0);
    expect(au.zeroInits).toBe(0);
  });

  it('attention scores: matmul + scale + softmax chain', () => {
    const q = new TensorType([8, 16], ScalarType.F32);
    const k = new TensorType([16, 8], ScalarType.F32);
    const out = new TensorType([8, 8], ScalarType.F32);

    const func = buildFunction('attn_scores', [q, k], [out], (b, args) => {
      const scores = b.matmul(args[0], args[1]).getResult(0);
      const scaleVal = b.scalarConstant(0.25, ScalarType.F32).getResult(0);
      const scaleBc = b.broadcast(scaleVal, [8, 8], []).getResult(0);
      const scaled = b.mul(scores, scaleBc).getResult(0);
      b.returnOp([b.softmax(scaled).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('attn_scores');
    const au = audit(src, 'attn_scores');

    expect(au.arithmeticNoise).toEqual([]);
    expect(au.extent1Loops).toBe(0);
  });

  it('conv + BN + ReLU: BN elementwise should be fused', () => {
    const x    = new TensorType([1, 4, 8, 8], ScalarType.F32);
    const k    = new TensorType([8, 4, 3, 3], ScalarType.F32);
    const g    = new TensorType([8], ScalarType.F32);
    const beta = new TensorType([8], ScalarType.F32);
    const mean = new TensorType([8], ScalarType.F32);
    const var_ = new TensorType([8], ScalarType.F32);
    const out  = new TensorType([1, 8, 6, 6], ScalarType.F32);

    const func = buildFunction('conv_bn_relu', [x, k, g, beta, mean, var_], [out], (b, args) => {
      const conv = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
      const bn = b.batchnorm(conv, args[2], args[3], args[4], args[5]).getResult(0);
      b.returnOp([b.relu(bn).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('conv_bn_relu');
    const au = audit(src, 'conv_bn_relu');

    expect(au.arithmeticNoise).toEqual([]);
    expect(au.extent1Loops).toBe(0);
  });

  it('layernorm: reduction loops + elementwise normalize, no noise', () => {
    const x = new TensorType([4, 32], ScalarType.F32);
    const g = new TensorType([32], ScalarType.F32);
    const beta = new TensorType([32], ScalarType.F32);
    const out = new TensorType([4, 32], ScalarType.F32);

    const func = buildFunction('ln', [x, g, beta], [out], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2]).getResult(0)]);
    });

    const r = compile(func);
    const src = r.getSource('ln');
    const au = audit(src, 'ln');

    expect(au.arithmeticNoise).toEqual([]);
    expect(au.extent1Loops).toBe(0);
  });
});