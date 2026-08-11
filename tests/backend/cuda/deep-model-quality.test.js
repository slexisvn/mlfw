import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { CUDATarget } from '../../../src/backend/target.js';
import { F32 } from '../../_utils/ir_fixture.js';
import {
  tensor, Linear, Sequential, ReLU, Sigmoid, Tanh,
  GELU, SiLU, LeakyReLU,
  compile as modelCompile,
} from '../../../src/index.js';

const F64 = ScalarType.F64;

function t(shape, dtype = F32) { return new TensorType(shape, dtype); }

function compile(func, opts = {}) {
  return compileGraph(func, CUDATarget(), { scheduling: { enabled: true }, ...opts });
}

function compileGPU(model, inputs, opts) {
  return modelCompile(model, inputs, { target: CUDATarget(), ...opts });
}

function modelAllSrc(compiled) {
  const mod = compiled.result().module;
  return mod.listKernels().map(n => mod.getKernelSource(n)).join('\n');
}

function modelKernelCount(compiled) {
  return compiled.result().module.listKernels().length;
}

function getSource(result, name) { return result.getSource(name); }

function getAllSource(result) {
  return result.listKernels().map(k => result.getSource(k)).join('\n');
}

function extractDeclaredVars(src) {
  const declared = new Set();
  const paramMatch = src.match(/__global__\s+void\s+\w+\(([^)]*)\)/);
  if (paramMatch) {
    for (const p of paramMatch[1].split(',')) {
      const name = p.trim().split(/\s+/).pop().replace('*', '');
      if (name) declared.add(name);
    }
  }
  for (const m of src.matchAll(/(?:const\s+int|int|float|double|__half)\s+(\w+)\s*(?:=|\[)/g)) {
    declared.add(m[1]);
  }
  for (const m of src.matchAll(/for\s*\(\s*int\s+(\w+)/g)) {
    declared.add(m[1]);
  }
  declared.add('blockIdx');
  declared.add('threadIdx');
  declared.add('INFINITY');
  return declared;
}

function findUndeclaredVars(src) {
  const declared = extractDeclaredVars(src);
  const bodyMatch = src.match(/\{([\s\S]*)\}\s*$/);
  if (!bodyMatch) return [];
  const body = bodyMatch[1];
  const used = new Set();
  for (const m of body.matchAll(/\b([a-zA-Z_]\w*)\b/g)) {
    const name = m[1];
    if (/^(const|int|float|double|void|for|if|else|while|return|__global__|__shared__|__half|__syncthreads|pragma|unroll|INFINITY|alloca|sizeof)$/.test(name)) continue;
    if (/^(expf|logf|sqrtf|tanhf|fabsf|sinf|cosf|ceilf|floorf|fmaxf|fminf|powf|roundf|fmodf|rsqrtf|rsqrt|exp|log|sqrt|tanh|fabs|sin|cos|ceil|floor|fmax|fmin|pow|round|fmod)$/.test(name)) continue;
    if (name.length <= 1 && /^[xyzw]$/.test(name)) {
      const before = m.index > 0 ? body[m.index - 1] : '';
      if (before === '.') continue;
    }
    used.add(name);
  }
  const undeclared = [];
  for (const v of used) {
    if (!declared.has(v)) undeclared.push(v);
  }
  return undeclared;
}

function countStores(src) {
  return (src.match(/\w+\[.*\]\s*=/g) || []).length;
}

function hasNoopStore(src) {
  const bodyMatch = src.match(/\{([\s\S]*)\}\s*$/);
  if (!bodyMatch) return false;
  const lines = bodyMatch[1].split('\n').map(l => l.trim()).filter(l => l.length > 0);
  for (const line of lines) {
    const m = line.match(/^(\w+\[[^\]]+\])\s*=\s*(.+);$/);
    if (m) {
      const lhs = m[1].replace(/\s+/g, '');
      const rhs = m[2].replace(/\s+/g, '');
      if (lhs === rhs) return true;
    }
  }
  return false;
}

function checkKernelSanity(src) {
  expect(src).toMatch(/__global__/);
  expect(src).not.toMatch(/Math\./);
  expect(src).not.toMatch(/\blet\s/);
  expect(src).not.toMatch(/\bfunction\s/);
  expect(src).not.toMatch(/=>/);
  expect(src).not.toContain('undefined');
  expect(src).not.toContain('NaN');
  expect(src).not.toContain('[object');
  expect((src.match(/\{/g) || []).length).toBe((src.match(/\}/g) || []).length);
  expect((src.match(/\(/g) || []).length).toBe((src.match(/\)/g) || []).length);
}

describe('Deep MLP kernel quality', () => {
  it('6-layer MLP with relu: compiles, no undeclared vars, no JS artifacts', () => {
    const func = buildFunction('mlp_relu6', [t([8, 16])], [t([8, 8])], (b, args) => {
      let x = args[0];
      const sizes = [16, 32, 32, 16, 16, 8];
      let inSize = 16;
      for (const outSize of sizes) {
        const w = b.constant(0.01, t([inSize, outSize])).getResult(0);
        const bias = b.constant(0, t([8, outSize])).getResult(0);
        x = b.add(b.matmul(x, w).getResult(0), bias).getResult(0);
        x = b.relu(x).getResult(0);
        inSize = outSize;
      }
      b.returnOp([x]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('6-layer MLP with tanh: compiles with tanhf calls', () => {
    const func = buildFunction('mlp_tanh6', [t([8, 16])], [t([8, 8])], (b, args) => {
      let x = args[0];
      const sizes = [16, 32, 32, 16, 16, 8];
      let inSize = 16;
      for (const outSize of sizes) {
        const w = b.constant(0.01, t([inSize, outSize])).getResult(0);
        const bias = b.constant(0, t([8, outSize])).getResult(0);
        x = b.add(b.matmul(x, w).getResult(0), bias).getResult(0);
        x = b.tanh(x).getResult(0);
        inSize = outSize;
      }
      b.returnOp([x]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const allSrc = getAllSource(result);
    expect(allSrc).toContain('tanhf(');
    for (const k of result.listKernels()) {
      checkKernelSanity(getSource(result, k));
    }
  });

  it('4-layer MLP with sigmoid: compiles after epilogue fix', () => {
    const func = buildFunction('mlp_sig4', [t([16, 32])], [t([16, 8])], (b, args) => {
      let x = args[0];
      const sizes = [64, 32, 16, 8];
      let inSize = 32;
      for (const outSize of sizes) {
        const w = b.constant(0.01, t([inSize, outSize])).getResult(0);
        const bias = b.constant(0, t([16, outSize])).getResult(0);
        x = b.add(b.matmul(x, w).getResult(0), bias).getResult(0);
        x = b.sigmoid(x).getResult(0);
        inSize = outSize;
      }
      b.returnOp([x]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('MLP with mixed activations: relu, tanh, sigmoid, exp', () => {
    const func = buildFunction('mlp_mixed', [t([8, 16])], [t([8, 4])], (b, args) => {
      let x = args[0];
      const w1 = b.constant(0.01, t([16, 32])).getResult(0);
      x = b.relu(b.matmul(x, w1).getResult(0)).getResult(0);
      const w2 = b.constant(0.01, t([32, 16])).getResult(0);
      x = b.tanh(b.matmul(x, w2).getResult(0)).getResult(0);
      const w3 = b.constant(0.01, t([16, 8])).getResult(0);
      x = b.sigmoid(b.matmul(x, w3).getResult(0)).getResult(0);
      const w4 = b.constant(0.01, t([8, 4])).getResult(0);
      x = b.exp(b.matmul(x, w4).getResult(0)).getResult(0);
      b.returnOp([x]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Multi-head attention kernel quality', () => {
  it('QKV projection + scaled dot attention compiles correctly', () => {
    const func = buildFunction('mha', [t([4, 64]), t([64, 64]), t([64, 64]), t([64, 64])], [t([4, 64])], (b, args) => {
      const [x, wq, wk, wv] = args;
      const Q = b.matmul(x, wq).getResult(0);
      const K = b.matmul(x, wk).getResult(0);
      const V = b.matmul(x, wv).getResult(0);
      const Kt = b.transpose(K, [1, 0]).getResult(0);
      const scores = b.matmul(Q, Kt).getResult(0);
      const scale = b.constant(0.125, t([4, 4])).getResult(0);
      const scaled = b.mul(scores, scale).getResult(0);
      const attn = b.softmax(scaled).getResult(0);
      const out = b.matmul(attn, V).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('attention with bias and relu produces valid kernels', () => {
    const func = buildFunction('mha_biased', [t([8, 32]), t([32, 32]), t([32, 32]), t([8, 8])], [t([8, 8])], (b, args) => {
      const [x, wq, wk, bias] = args;
      const Q = b.matmul(x, wq).getResult(0);
      const K = b.matmul(x, wk).getResult(0);
      const Kt = b.transpose(K, [1, 0]).getResult(0);
      const scores = b.matmul(Q, Kt).getResult(0);
      const biased = b.add(scores, bias).getResult(0);
      const out = b.relu(biased).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('ResNet-style residual kernel quality', () => {
  it('matmul chain with skip connection compiles correctly', () => {
    const func = buildFunction('resblock', [t([16, 32]), t([32, 32]), t([32, 32])], [t([16, 32])], (b, args) => {
      const [x, w1, w2] = args;
      const h1 = b.relu(b.matmul(x, w1).getResult(0)).getResult(0);
      const h2 = b.matmul(h1, w2).getResult(0);
      const out = b.relu(b.add(h2, x).getResult(0)).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('double residual block (pre-activation ResNet style)', () => {
    const func = buildFunction('double_resblock', [t([8, 64]), t([64, 64]), t([64, 64]), t([64, 64]), t([64, 64])],
      [t([8, 64])], (b, args) => {
        const [x, w1, w2, w3, w4] = args;
        const h1 = b.matmul(b.relu(x).getResult(0), w1).getResult(0);
        const h2 = b.matmul(b.relu(h1).getResult(0), w2).getResult(0);
        const r1 = b.add(h2, x).getResult(0);
        const h3 = b.matmul(b.relu(r1).getResult(0), w3).getResult(0);
        const h4 = b.matmul(b.relu(h3).getResult(0), w4).getResult(0);
        const r2 = b.add(h4, r1).getResult(0);
        b.returnOp([r2]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Transformer FFN kernel quality', () => {
  it('layer norm + 2-layer FFN with relu compiles', () => {
    const func = buildFunction('ffn_relu', [t([4, 64]), t([64]), t([64]), t([64, 256]), t([256, 64])],
      [t([4, 64])], (b, args) => {
        const [x, gamma, beta, w1, w2] = args;
        const normed = b.layernorm(x, gamma, beta).getResult(0);
        const h = b.relu(b.matmul(normed, w1).getResult(0)).getResult(0);
        const out = b.matmul(h, w2).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('FFN with GELU activation (sigmoid-based) compiles', () => {
    const func = buildFunction('ffn_gelu', [t([4, 32]), t([32, 128]), t([128, 32])],
      [t([4, 32])], (b, args) => {
        const [x, w1, w2] = args;
        const h = b.matmul(x, w1).getResult(0);
        const activated = b.mul(h, b.sigmoid(h).getResult(0)).getResult(0);
        const out = b.matmul(activated, w2).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('FFN with residual + layer norm compiles', () => {
    const func = buildFunction('ffn_res_ln', [t([8, 64]), t([64]), t([64]), t([64, 256]), t([256, 64]), t([64]), t([64])],
      [t([8, 64])], (b, args) => {
        const [x, g1, b1, w1, w2, g2, b2] = args;
        const normed = b.layernorm(x, g1, b1).getResult(0);
        const h = b.relu(b.matmul(normed, w1).getResult(0)).getResult(0);
        const proj = b.matmul(h, w2).getResult(0);
        const residual = b.add(proj, x).getResult(0);
        const out = b.layernorm(residual, g2, b2).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Reduction chain kernel quality', () => {
  it('softmax + log + sum reduction compiles', () => {
    const func = buildFunction('xent', [t([16, 32])], [t([16])], (b, args) => {
      const probs = b.softmax(args[0]).getResult(0);
      const logp = b.log(probs).getResult(0);
      const zero = b.scalarConstant(0, F32).getResult(0);
      const out = b.reduce(logp, zero, [1], 'sum').getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('chained reductions (mean of mean) compiles', () => {
    const func = buildFunction('mean2', [t([8, 16, 32])], [t([8])], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      const r1 = b.reduce(args[0], zero, [2], 'sum').getResult(0);
      const r2 = b.reduce(r1, zero, [1], 'sum').getResult(0);
      b.returnOp([r2]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('max reduction + exp (log-sum-exp pattern)', () => {
    const func = buildFunction('logsumexp', [t([16, 64])], [t([16])], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, F32).getResult(0);
      const maxVal = b.reduce(args[0], negInf, [1], 'max').getResult(0);
      const maxBroadcast = b.broadcast(maxVal, [16, 64], [0]).getResult(0);
      const shifted = b.sub(args[0], maxBroadcast).getResult(0);
      const expd = b.exp(shifted).getResult(0);
      const zero = b.scalarConstant(0, F32).getResult(0);
      const sumExp = b.reduce(expd, zero, [1], 'sum').getResult(0);
      const logSum = b.log(sumExp).getResult(0);
      const out = b.add(logSum, maxVal).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Multi-output kernel quality', () => {
  it('encoder-decoder with 2 outputs compiles', () => {
    const func = buildFunction('enc_dec', [t([8, 32]), t([32, 16]), t([16, 32]), t([16, 4])],
      [t([8, 32]), t([8, 4])], (b, args) => {
        const [x, wEnc, wDec, wCls] = args;
        const enc = b.relu(b.matmul(x, wEnc).getResult(0)).getResult(0);
        const dec = b.tanh(b.matmul(enc, wDec).getResult(0)).getResult(0);
        const cls = b.matmul(enc, wCls).getResult(0);
        b.returnOp([dec, cls]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('3-head output model compiles', () => {
    const func = buildFunction('multi3', [t([4, 16]), t([16, 32]), t([32, 8]), t([32, 4]), t([32, 2])],
      [t([4, 8]), t([4, 4]), t([4, 2])], (b, args) => {
        const [x, wShared, w1, w2, w3] = args;
        const shared = b.relu(b.matmul(x, wShared).getResult(0)).getResult(0);
        const o1 = b.matmul(shared, w1).getResult(0);
        const o2 = b.matmul(shared, w2).getResult(0);
        const o3 = b.matmul(shared, w3).getResult(0);
        b.returnOp([o1, o2, o3]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Deep elementwise chain kernel quality', () => {
  it('10-deep unary chain with various activations', () => {
    const func = buildFunction('unary10', [t([512])], [t([512])], (b, args) => {
      let v = args[0];
      const ops = [
        x => b.exp(x).getResult(0),
        x => b.tanh(x).getResult(0),
        x => b.sigmoid(x).getResult(0),
        x => b.relu(x).getResult(0),
        x => b.neg(x).getResult(0),
      ];
      for (let i = 0; i < 10; i++) {
        v = ops[i % ops.length](v);
      }
      b.returnOp([v]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(countStores(src)).toBeLessThanOrEqual(3);
  });

  it('8-deep mixed binary+unary chain', () => {
    const func = buildFunction('mixed8', [t([512]), t([512])], [t([512])], (b, args) => {
      let v = b.add(args[0], args[1]).getResult(0);
      v = b.exp(v).getResult(0);
      v = b.mul(v, args[0]).getResult(0);
      v = b.tanh(v).getResult(0);
      v = b.sub(v, args[1]).getResult(0);
      v = b.sigmoid(v).getResult(0);
      v = b.neg(v).getResult(0);
      v = b.sqrt(b.mul(v, v).getResult(0)).getResult(0);
      b.returnOp([v]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(countStores(src)).toBeLessThanOrEqual(2);
  });

  it('15-deep pure unary chain (no sigmoid) fuses to 1 store', () => {
    const func = buildFunction('unary15', [t([256])], [t([256])], (b, args) => {
      let v = args[0];
      const ops = [
        x => b.exp(x).getResult(0),
        x => b.tanh(x).getResult(0),
        x => b.neg(x).getResult(0),
        x => b.abs(x).getResult(0),
        x => b.sqrt(x).getResult(0),
      ];
      for (let i = 0; i < 15; i++) {
        v = ops[i % ops.length](v);
      }
      b.returnOp([v]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(countStores(src)).toBe(1);
  });

  it('6-deep binary chain fuses to 1 store', () => {
    const func = buildFunction('bin6', [t([256]), t([256]), t([256])], [t([256])], (b, args) => {
      let v = b.add(args[0], args[1]).getResult(0);
      v = b.mul(v, args[2]).getResult(0);
      v = b.sub(v, args[0]).getResult(0);
      v = b.add(v, args[1]).getResult(0);
      v = b.mul(v, args[2]).getResult(0);
      v = b.sub(v, args[0]).getResult(0);
      b.returnOp([v]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(countStores(src)).toBe(1);
  });
});

describe('Shape manipulation kernel quality', () => {
  it('reshape -> matmul -> transpose -> reduce compiles', () => {
    const func = buildFunction('shape_ops', [t([4, 8, 2]), t([16, 16])], [t([4])], (b, args) => {
      const reshaped = b.reshape(args[0], [4, 16]).getResult(0);
      const mm = b.matmul(reshaped, args[1]).getResult(0);
      const transposed = b.transpose(mm, [1, 0]).getResult(0);
      const zero = b.scalarConstant(0, F32).getResult(0);
      const out = b.reduce(transposed, zero, [0], 'sum').getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('broadcast + elementwise chain compiles', () => {
    const func = buildFunction('bcast_chain', [t([8, 1]), t([1, 16])], [t([8, 16])], (b, args) => {
      const a = b.broadcast(args[0], [8, 16], [0, 1]).getResult(0);
      const c = b.broadcast(args[1], [8, 16], [0, 1]).getResult(0);
      const sum = b.add(a, c).getResult(0);
      const out = b.relu(sum).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('slice + concat-like pattern compiles', () => {
    const func = buildFunction('slice_ops', [t([16, 32])], [t([8, 16])], (b, args) => {
      const sliced = b.slice(args[0], [0, 0], [8, 16]).getResult(0);
      const out = b.relu(sliced).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Large matmul kernel quality', () => {
  it('256x256 matmul compiles with valid tiling', () => {
    const func = buildFunction('mm256', [t([256, 256]), t([256, 256])], [t([256, 256])], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/blockIdx/);
    expect(src).toMatch(/threadIdx/);
  });

  it('non-square matmul 64x128 @ 128x32 compiles', () => {
    const func = buildFunction('mm_nonsq', [t([64, 128]), t([128, 32])], [t([64, 32])], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src.includes('__shared__')).toBe(true);
  });

  it('tall-skinny matmul 512x4 @ 4x8 compiles', () => {
    const func = buildFunction('mm_tall', [t([512, 4]), t([4, 8])], [t([512, 8])], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

describe('Dtype correctness in deep models', () => {
  it('f64 MLP uses double* params and unsuffixed math', () => {
    const func = buildFunction('mlp_f64', [t([8, 16], F64), t([16, 8], F64)], [t([8, 8], F64)], (b, args) => {
      const h = b.matmul(args[0], args[1]).getResult(0);
      const out = b.exp(h).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(src).toMatch(/double\*/);
    expect(src).toMatch(/\bexp\(/);
    expect(src).not.toMatch(/expf\(/);
  });

  it('f32 MLP uses float* params and f-suffixed math', () => {
    const func = buildFunction('mlp_f32', [t([8, 16]), t([16, 8])], [t([8, 8])], (b, args) => {
      const h = b.matmul(args[0], args[1]).getResult(0);
      const out = b.tanh(h).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/float\*/);
    expect(src).toMatch(/tanhf\(/);
    expect(src).not.toMatch(/double\*/);
  });
});

describe('Broadcasting kernel quality', () => {
  it('matmul + broadcast bias add + relu compiles', () => {
    const func = buildFunction('mm_bcast_bias', [t([16, 32]), t([32, 64]), t([64])], [t([16, 64])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const bias = b.broadcast(args[2], [16, 64], [1]).getResult(0);
      const biased = b.add(mm, bias).getResult(0);
      const out = b.relu(biased).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('scalar broadcast + elementwise chain compiles', () => {
    const func = buildFunction('scalar_bcast', [t([32, 64])], [t([32, 64])], (b, args) => {
      const scale = b.scalarConstant(0.5, F32).getResult(0);
      const scaleBcast = b.broadcast(scale, [32, 64], []).getResult(0);
      const out = b.mul(args[0], scaleBcast).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Pooling kernel quality', () => {
  it('matmul + reduce (mean pooling) compiles', () => {
    const func = buildFunction('mm_pool',
      [t([8, 32]), t([32, 64])],
      [t([8])],
      (b, args) => {
        const projected = b.matmul(args[0], args[1]).getResult(0);
        const zero = b.scalarConstant(0, F32).getResult(0);
        const pooled = b.reduce(projected, zero, [1], 'sum').getResult(0);
        b.returnOp([pooled]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Matmul + sigmoid (epilogue regression)', () => {
  it('matmul + sigmoid compiles without crash', () => {
    const func = buildFunction('mm_sig', [t([32, 64]), t([64, 64])], [t([32, 64])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const out = b.sigmoid(mm).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('matmul + add + sigmoid compiles', () => {
    const func = buildFunction('mm_add_sig', [t([16, 32]), t([32, 16]), t([16, 16])], [t([16, 16])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const biased = b.add(mm, args[2]).getResult(0);
      const out = b.sigmoid(biased).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('sigmoid alone compiles to single kernel', () => {
    const func = buildFunction('sig_only', [t([256])], [t([256])], (b, args) => {
      b.returnOp([b.sigmoid(args[0]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/expf\(/);
  });

  it('matmul + sigmoid + matmul pipeline compiles', () => {
    const func = buildFunction('mm_sig_mm',
      [t([8, 16]), t([16, 32]), t([32, 8])],
      [t([8, 8])], (b, args) => {
        const h = b.sigmoid(b.matmul(args[0], args[1]).getResult(0)).getResult(0);
        const out = b.matmul(h, args[2]).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Epilogue fusion quality', () => {
  it('matmul + relu fuses into single kernel', () => {
    const func = buildFunction('mm_relu', [t([16, 32]), t([32, 16])], [t([16, 16])], (b, args) => {
      b.returnOp([b.relu(b.matmul(args[0], args[1]).getResult(0)).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/fmaxf\(/);
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('matmul + bias + tanh fuses into single kernel', () => {
    const func = buildFunction('mm_bias_tanh', [t([8, 16]), t([16, 8]), t([8, 8])], [t([8, 8])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const biased = b.add(mm, args[2]).getResult(0);
      b.returnOp([b.tanh(biased).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/tanhf\(/);
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('matmul + scale + neg fuses into single kernel', () => {
    const func = buildFunction('mm_scale_neg', [t([8, 16]), t([16, 8]), t([8, 8])], [t([8, 8])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const scaled = b.mul(mm, args[2]).getResult(0);
      b.returnOp([b.neg(scaled).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(hasNoopStore(src)).toBe(false);
  });

  it('matmul + bias + relu + scale: 4-op epilogue chain', () => {
    const func = buildFunction('mm_4epi', [t([8, 16]), t([16, 8]), t([8, 8]), t([8, 8])], [t([8, 8])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const biased = b.add(mm, args[2]).getResult(0);
      const activated = b.relu(biased).getResult(0);
      b.returnOp([b.mul(activated, args[3]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    expect(src).toMatch(/fmaxf\(/);
    expect(findUndeclaredVars(src)).toEqual([]);
  });
});

describe('Conv kernel quality', () => {
  it('2D conv compiles with valid CUDA', () => {
    const func = buildFunction('conv2d',
      [t([1, 3, 16, 16]), t([8, 3, 3, 3])],
      [t([1, 8, 14, 14])], (b, args) => {
        b.returnOp([b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0)]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('conv + relu + conv chain compiles', () => {
    const func = buildFunction('conv_relu_conv',
      [t([1, 3, 16, 16]), t([8, 3, 3, 3]), t([8, 8, 3, 3])],
      [t([1, 8, 12, 12])], (b, args) => {
        const c1 = b.conv(args[0], args[1], [1, 1], [[0, 0], [0, 0]]).getResult(0);
        const r1 = b.relu(c1).getResult(0);
        const c2 = b.conv(r1, args[2], [1, 1], [[0, 0], [0, 0]]).getResult(0);
        b.returnOp([c2]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('conv with padding compiles with bounds guard', () => {
    const func = buildFunction('conv_pad',
      [t([1, 3, 8, 8]), t([4, 3, 3, 3])],
      [t([1, 4, 8, 8])], (b, args) => {
        b.returnOp([b.conv(args[0], args[1], [1, 1], [[1, 1], [1, 1]]).getResult(0)]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const allSrc = getAllSource(result);
    const hasBoundsGuard = /if\s*\(/.test(allSrc) || /\?\s*\(/.test(allSrc) || />=\s*0/.test(allSrc);
    expect(hasBoundsGuard).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Model-level GPU compilation quality', () => {
  it('Sequential MLP with ReLU: valid CUDA, shared memory', () => {
    const model = new Sequential(
      new Linear(8, 32), new ReLU(),
      new Linear(32, 16), new ReLU(),
      new Linear(16, 4),
    );
    const x = tensor([Array.from({ length: 8 }, (_, i) => i * 0.1)]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = modelAllSrc(compiled);
    expect(modelKernelCount(compiled)).toBeGreaterThan(1);
    expect(src).toContain('__global__');
    expect(src).not.toMatch(/Math\./);
    expect(src).not.toContain('undefined');
  });

  it('Sequential with Tanh activation: tanhf in output', () => {
    const model = new Sequential(
      new Linear(4, 16), new Tanh(),
      new Linear(16, 4),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = modelAllSrc(compiled);
    expect(src).toContain('tanhf(');
    expect(src).toContain('__global__');
  });

  it('Sequential with Sigmoid: compiles after epilogue fix', () => {
    const model = new Sequential(
      new Linear(4, 8), new Sigmoid(),
      new Linear(8, 2),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = modelAllSrc(compiled);
    expect(src).toContain('__global__');
    expect(src).toContain('expf(');
    expect(src).not.toContain('undefined');
  });

  it('Sequential with GELU: compiles without crash', () => {
    const model = new Sequential(
      new Linear(4, 16), new GELU(),
      new Linear(16, 4),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = compiled.source();
    expect(src).toContain('__global__');
    expect(src).not.toContain('undefined');
  });

  it('Sequential with SiLU: compiles without crash', () => {
    const model = new Sequential(
      new Linear(4, 16), new SiLU(),
      new Linear(16, 4),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = compiled.source();
    expect(src).toContain('__global__');
    expect(src).not.toContain('undefined');
  });

  it('Sequential with LeakyReLU: compiles', () => {
    const model = new Sequential(
      new Linear(4, 16), new LeakyReLU(),
      new Linear(16, 4),
    );
    const x = tensor([[1, 2, 3, 4]]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = compiled.source();
    expect(src).toContain('__global__');
    expect(src).not.toContain('undefined');
  });

  it('deep 8-layer model scheduled with valid CUDA', () => {
    const model = new Sequential(
      new Linear(8, 32), new ReLU(),
      new Linear(32, 64), new ReLU(),
      new Linear(64, 32), new ReLU(),
      new Linear(32, 16), new ReLU(),
      new Linear(16, 8), new ReLU(),
      new Linear(8, 4), new ReLU(),
      new Linear(4, 2), new ReLU(),
      new Linear(2, 1),
    );
    const x = tensor([Array.from({ length: 8 }, (_, i) => i * 0.1)]);
    const compiled = compileGPU(model, [x], { scheduling: { enabled: true } });
    const src = modelAllSrc(compiled);
    expect(modelKernelCount(compiled)).toBeGreaterThan(1);
    expect(src).toContain('__global__');
    let depth = 0;
    for (const ch of src) {
      if (ch === '{') depth++;
      if (ch === '}') depth--;
    }
    expect(depth).toBe(0);
  });
});

describe('Numerical edge case kernel quality', () => {
  it('1-element tensors compile without issues', () => {
    const func = buildFunction('scalar_mm', [t([1, 1]), t([1, 1])], [t([1, 1])], (b, args) => {
      const mm = b.matmul(args[0], args[1]).getResult(0);
      const out = b.exp(mm).getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      expect(src).not.toContain('undefined');
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('non-power-of-2 matmul 13x7 @ 7x11 compiles', () => {
    const func = buildFunction('mm_odd', [t([13, 7]), t([7, 11])], [t([13, 11])], (b, args) => {
      b.returnOp([b.matmul(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
  });

  it('very wide vector 1x4096 elementwise compiles', () => {
    const func = buildFunction('wide_vec', [t([1, 4096]), t([1, 4096])], [t([1, 4096])], (b, args) => {
      const sum = b.add(args[0], args[1]).getResult(0);
      b.returnOp([b.relu(sum).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/blockIdx/);
  });

  it('reduction to scalar compiles', () => {
    const func = buildFunction('red_scalar', [t([64, 32])], [t([])], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      const r1 = b.reduce(args[0], zero, [1], 'sum').getResult(0);
      const out = b.reduce(r1, zero, [0], 'sum').getResult(0);
      b.returnOp([out]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Compound activation kernel quality', () => {
  it('softmax compiles to valid CUDA kernels', () => {
    const func = buildFunction('sm', [t([8, 32])], [t([8, 32])], (b, args) => {
      b.returnOp([b.softmax(args[0]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('layer norm compiles to valid CUDA kernels', () => {
    const func = buildFunction('ln', [t([4, 64]), t([64]), t([64])], [t([4, 64])], (b, args) => {
      b.returnOp([b.layernorm(args[0], args[1], args[2]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('softmax + matmul pipeline compiles', () => {
    const func = buildFunction('sm_mm', [t([8, 16]), t([16, 32])], [t([8, 32])], (b, args) => {
      const sm = b.softmax(args[0]).getResult(0);
      b.returnOp([b.matmul(sm, args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('layer norm + sigmoid compiles (both decompose)', () => {
    const func = buildFunction('ln_sig', [t([4, 32]), t([32]), t([32])], [t([4, 32])], (b, args) => {
      const normed = b.layernorm(args[0], args[1], args[2]).getResult(0);
      b.returnOp([b.sigmoid(normed).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('Comparison & select kernel quality', () => {
  it('maximum (element-wise max) compiles', () => {
    const func = buildFunction('ew_max', [t([256]), t([256])], [t([256])], (b, args) => {
      b.returnOp([b.maximum(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/fmaxf\(/);
  });

  it('minimum (element-wise min) compiles', () => {
    const func = buildFunction('ew_min', [t([256]), t([256])], [t([256])], (b, args) => {
      b.returnOp([b.minimum(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.listKernels()).toHaveLength(1);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(src).toMatch(/fminf\(/);
  });

  it('clamp compiles with min and max', () => {
    const func = buildFunction('clamp_op', [t([256])], [t([256])], (b, args) => {
      const lo = b.constant(-1, t([256])).getResult(0);
      const hi = b.constant(1, t([256])).getResult(0);
      b.returnOp([b.clamp(lo, args[0], hi).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('High-dimensional tensor kernel quality', () => {
  it('4D elementwise add compiles', () => {
    const func = buildFunction('add_4d', [t([2, 4, 8, 16]), t([2, 4, 8, 16])], [t([2, 4, 8, 16])], (b, args) => {
      b.returnOp([b.add(args[0], args[1]).getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    const src = getSource(result, result.listKernels()[0]);
    checkKernelSanity(src);
    expect(findUndeclaredVars(src)).toEqual([]);
    expect(countStores(src)).toBe(1);
  });

  it('3D reduction to 2D compiles', () => {
    const func = buildFunction('red_3d', [t([4, 8, 16])], [t([4, 8])], (b, args) => {
      const zero = b.scalarConstant(0, F32).getResult(0);
      b.returnOp([b.reduce(args[0], zero, [2], 'sum').getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('4D max reduction along axis 3 compiles', () => {
    const func = buildFunction('red_4d_max', [t([2, 4, 8, 16])], [t([2, 4, 8])], (b, args) => {
      const negInf = b.scalarConstant(-Infinity, F32).getResult(0);
      b.returnOp([b.reduce(args[0], negInf, [3], 'max').getResult(0)]);
    });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});

describe('End-to-end model pattern kernel quality', () => {
  it('autoencoder (encode + decode) compiles', () => {
    const func = buildFunction('autoenc',
      [t([4, 64]), t([64, 16]), t([16, 8]), t([8, 16]), t([16, 64])],
      [t([4, 64])], (b, args) => {
        const [x, w1, w2, w3, w4] = args;
        let h = b.relu(b.matmul(x, w1).getResult(0)).getResult(0);
        h = b.relu(b.matmul(h, w2).getResult(0)).getResult(0);
        h = b.relu(b.matmul(h, w3).getResult(0)).getResult(0);
        const out = b.matmul(h, w4).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('classifier (features + logits + softmax) compiles', () => {
    const func = buildFunction('classifier',
      [t([8, 32]), t([32, 64]), t([64, 10])],
      [t([8, 10])], (b, args) => {
        const [x, w1, w2] = args;
        const features = b.relu(b.matmul(x, w1).getResult(0)).getResult(0);
        const logits = b.matmul(features, w2).getResult(0);
        const probs = b.softmax(logits).getResult(0);
        b.returnOp([probs]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('GAN discriminator pattern compiles', () => {
    const func = buildFunction('disc',
      [t([4, 64]), t([64, 32]), t([32, 16]), t([16, 1])],
      [t([4, 1])], (b, args) => {
        const [x, w1, w2, w3] = args;
        let h = b.relu(b.matmul(x, w1).getResult(0)).getResult(0);
        h = b.relu(b.matmul(h, w2).getResult(0)).getResult(0);
        const logit = b.matmul(h, w3).getResult(0);
        const out = b.sigmoid(logit).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });

  it('batch normalization pattern compiles', () => {
    const func = buildFunction('batchnorm_manual',
      [t([8, 64]), t([64]), t([64]), t([64]), t([64])],
      [t([8, 64])], (b, args) => {
        const [x, gamma, beta, mean, variance] = args;
        const meanBcast = b.broadcast(mean, [8, 64], [1]).getResult(0);
        const varBcast = b.broadcast(variance, [8, 64], [1]).getResult(0);
        const centered = b.sub(x, meanBcast).getResult(0);
        const eps = b.constant(1e-5, t([8, 64])).getResult(0);
        const denom = b.sqrt(b.add(varBcast, eps).getResult(0)).getResult(0);
        const normed = b.div(centered, denom).getResult(0);
        const gammaBcast = b.broadcast(gamma, [8, 64], [1]).getResult(0);
        const betaBcast = b.broadcast(beta, [8, 64], [1]).getResult(0);
        const out = b.add(b.mul(normed, gammaBcast).getResult(0), betaBcast).getResult(0);
        b.returnOp([out]);
      });
    const result = compile(func);
    expect(result.succeeded).toBe(true);
    for (const k of result.listKernels()) {
      const src = getSource(result, k);
      checkKernelSanity(src);
      expect(findUndeclaredVars(src)).toEqual([]);
    }
  });
});
