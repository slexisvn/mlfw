import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/compiler/support/target.js';
import { F32 } from '../_utils/ir_fixture.js';

const target = CPUTarget();
const tt = (shape) => new TensorType(shape, F32);

const compileBaseline = (func) => compileGraph(func, target);
const compileWithAutotune = (func, tuneOpts = {}) =>
  compileGraph(func, target, { scheduling: { enabled: true, autotune: true, ...tuneOpts } });

function countOps(func) {
  let n = 0;
  for (const _op of func.ops()) n++;
  return n;
}

function analyzeKernel(src) {
  const issues = [];
  if (/\(0\s*-\s*\w/.test(src)) issues.push('NEG_NOISE');
  if (/\*\s*1\b(?!\.)/.test(src)) issues.push('MUL_ONE');
  return { loops: (src.match(/\bfor\s*\(/g) || []).length, lines: src.split('\n').length, issues };
}

function report(label, ops, baseMs, tuneMs, baseK, tuneK) {
  const quality = (k) => (k.issues.length === 0 ? 'CLEAN' : k.issues.join(','));
  const overhead = baseMs > 0 ? (tuneMs / baseMs).toFixed(1) : '?';
  console.log(
    `  [${label}] graphOps=${ops}\n` +
    `    baseline: ${baseMs.toFixed(0)}ms loops=${baseK.loops} lines=${baseK.lines} quality=${quality(baseK)}\n` +
    `    autotune: ${tuneMs.toFixed(0)}ms loops=${tuneK.loops} lines=${tuneK.lines} quality=${quality(tuneK)}\n` +
    `    compile overhead: ${overhead}x`
  );
}

function attention(b, x, seq, d, wQ, wK, wV, wO) {
  const q = b.matmul(x, wQ).getResult(0);
  const k = b.matmul(x, wK).getResult(0);
  const v = b.matmul(x, wV).getResult(0);
  const scores = b.matmul(q, b.transpose(k, [1, 0]).getResult(0)).getResult(0);
  const scale = b.broadcast(b.scalarConstant(1 / Math.sqrt(d), F32).getResult(0), [seq, seq], []).getResult(0);
  const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
  return b.matmul(b.matmul(attn, v).getResult(0), wO).getResult(0);
}

const normed = (b, x, seq, d, gain, bias) =>
  b.reshape(b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), gain, bias, 2).getResult(0), [seq, d]).getResult(0);

function buildResidualConvNet(name, { layers, channels, size }) {
  const shape = tt([1, channels, size, size]);
  return buildFunction(name, [shape], [shape], (b, args) => {
    let x = args[0];
    for (let i = 0; i < layers; i++) {
      const w = b.constant(0.01, tt([channels, channels, 3, 3])).getResult(0);
      const gain = b.constant(1.0, tt([channels])).getResult(0);
      const bias = b.constant(0.0, tt([channels])).getResult(0);
      const mean = b.constant(0.0, tt([channels])).getResult(0);
      const variance = b.constant(1.0, tt([channels])).getResult(0);
      const conv = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]).getResult(0);
      const bn = b.batchnorm(conv, gain, bias, mean, variance).getResult(0);
      x = b.relu(b.add(bn, x).getResult(0)).getResult(0);
    }
    b.returnOp([x]);
  });
}

function buildPreNormTransformer(name, { layers, seq, d, ffMult }) {
  const params = [];
  for (let i = 0; i < layers; i++) {
    params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]), tt([d]), tt([d]),
      tt([d, d * ffMult]), tt([d * ffMult, d]), tt([d]), tt([d]));
  }
  return buildFunction(name, [tt([seq, d]), ...params], [tt([seq, d])], (b, args) => {
    let x = args[0];
    let pi = 1;
    for (let i = 0; i < layers; i++) {
      const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
      const gain1 = args[pi++], bias1 = args[pi++];
      const wF1 = args[pi++], wF2 = args[pi++];
      const gain2 = args[pi++], bias2 = args[pi++];
      x = b.add(x, attention(b, normed(b, x, seq, d, gain1, bias1), seq, d, wQ, wK, wV, wO)).getResult(0);
      const hidden = b.gelu(b.matmul(normed(b, x, seq, d, gain2, bias2), wF1).getResult(0)).getResult(0);
      x = b.add(x, b.matmul(hidden, wF2).getResult(0)).getResult(0);
    }
    b.returnOp([x]);
  });
}

function buildPostNormAttentionStack(name, { layers, seq, d }) {
  const params = [];
  for (let i = 0; i < layers; i++) {
    params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]), tt([d]), tt([d]));
  }
  return buildFunction(name, [tt([seq, d]), ...params], [tt([seq, d])], (b, args) => {
    let x = args[0];
    let pi = 1;
    for (let i = 0; i < layers; i++) {
      const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
      const gain = args[pi++], bias = args[pi++];
      x = b.add(x, attention(b, x, seq, d, wQ, wK, wV, wO)).getResult(0);
      x = normed(b, x, seq, d, gain, bias);
    }
    b.returnOp([x]);
  });
}

const SCALE_MODELS = [
  { label: 'ResNet 5 blocks', build: (n) => buildResidualConvNet(n, { layers: 5, channels: 32, size: 8 }) },
  { label: 'Transformer 6L', build: (n) => buildPreNormTransformer(n, { layers: 6, seq: 8, d: 32, ffMult: 2 }) },
  { label: 'GPT 24L', build: (n) => buildPreNormTransformer(n, { layers: 24, seq: 8, d: 32, ffMult: 2 }) },
  { label: 'XL 50L', build: (n) => buildPostNormAttentionStack(n, { layers: 50, seq: 4, d: 16 }) },
  { label: 'Mega 100L', build: (n) => buildPostNormAttentionStack(n, { layers: 100, seq: 4, d: 16 }) },
  { label: 'Monster 200L', build: (n) => buildPostNormAttentionStack(n, { layers: 200, seq: 4, d: 16 }), minLoops: 5000 },
];

describe('autotune stress — compile time & kernel quality at scale', { timeout: 600000 }, () => {
  it.each(SCALE_MODELS)('$label: autotune preserves kernel quality', ({ label, build, minLoops }) => {
    const slug = label.toLowerCase().replace(/[^a-z0-9]+/g, '_');
    const baseName = `${slug}_base`, tuneName = `${slug}_tune`;
    const baseFunc = build(baseName);
    const ops = countOps(baseFunc);

    const t0 = performance.now();
    const rBase = compileBaseline(baseFunc);
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    const rTune = compileWithAutotune(build(tuneName));
    const tuneMs = performance.now() - t1;

    const baseK = analyzeKernel(rBase.getSource(baseName));
    const tuneK = analyzeKernel(rTune.getSource(tuneName));
    report(label, ops, baseMs, tuneMs, baseK, tuneK);

    expect(baseK.issues, `${label} baseline kernel quality`).toHaveLength(0);
    expect(tuneK.issues, `${label} autotuned kernel quality`).toHaveLength(0);
    if (minLoops !== undefined) {
      expect(tuneK.loops, `${label} autotuned loop count`).toBeGreaterThan(minLoops);
    }
  });

  it('correctness: autotuned kernel produces same output as baseline', () => {
    const seq = 4, d = 8;
    const buildModel = (name) => buildFunction(name, [tt([seq, d]), tt([d, d]), tt([d, d])], [tt([seq, d])], (b, args) => {
      const x = args[0];
      const q = b.matmul(x, args[1]).getResult(0);
      const k = b.matmul(x, args[2]).getResult(0);
      const scores = b.matmul(q, b.transpose(k, [1, 0]).getResult(0)).getResult(0);
      const attn = b.softmax(scores, 1).getResult(0);
      const out = b.matmul(attn, x).getResult(0);
      b.returnOp([b.relu(b.add(x, out).getResult(0)).getResult(0)]);
    });

    const rBase = compileBaseline(buildModel('corr_base'));
    const rTune = compileWithAutotune(buildModel('corr_tune'));

    const fill = (n, f) => Float32Array.from({ length: n }, (_, i) => f(i));
    const inp = fill(seq * d, (i) => (i % 7 - 3) * 0.1);
    const w1 = fill(d * d, (i) => (i % 5 - 2) * 0.05);
    const w2 = fill(d * d, (i) => (i % 3 - 1) * 0.1);

    const outBase = new Float32Array(seq * d), outTune = new Float32Array(seq * d);
    rBase.run('corr_base', inp, w1, w2, outBase);
    rTune.run('corr_tune', inp, w1, w2, outTune);

    for (let i = 0; i < outBase.length; i++) {
      expect(Math.abs(outBase[i] - outTune[i]), `element ${i} matches baseline`).toBeLessThan(1e-5);
    }
  });
});
