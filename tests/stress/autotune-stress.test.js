import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/backend/target.js';

const F32 = ScalarType.F32;
const target = CPUTarget();
function tt(shape) { return new TensorType(shape, F32); }

function compileWithAutotune(func, tuneOpts = {}) {
  return compileGraph(func, target, {
    scheduling: { enabled: true, autotune: true, ...tuneOpts },
  });
}

function compileBaseline(func) {
  return compileGraph(func, target);
}

function analyzeKernel(src) {
  const loops = (src.match(/\bfor\s*\(/g) || []).length;
  const temps = (src.match(/\bnew Float32Array\(/g) || []).length;
  const lines = src.split('\n').length;
  const hasParallel = /\/\/ parallel/.test(src) || /\.parallel/.test(src);
  const hasSplit = src.includes('/*split*/') || (src.match(/\bfor\s*\(/g) || []).length > loops;
  const issues = [];
  if (/\(0\s*-\s*\w/.test(src)) issues.push('NEG_NOISE');
  if (/\*\s*1\b(?!\.)/.test(src)) issues.push('MUL_ONE');
  if (/\+\s*0(?!\.\d*[1-9])\b/.test(src)) issues.push('ADD_ZERO');
  return { loops, temps, lines, issues };
}

function countOps(func) {
  let count = 0;
  for (const op of func.ops()) count++;
  return count;
}

function log(name, graphOps, baseMs, tuneMs, baseK, tuneK) {
  const speedup = baseMs > 0 ? (tuneMs / baseMs).toFixed(1) : '?';
  const qBase = baseK.issues.length === 0 ? 'CLEAN' : baseK.issues.join(',');
  const qTune = tuneK.issues.length === 0 ? 'CLEAN' : tuneK.issues.join(',');
  console.log(
    `  [${name}] graphOps=${graphOps}\n` +
    `    baseline: ${baseMs.toFixed(0)}ms loops=${baseK.loops} lines=${baseK.lines} quality=${qBase}\n` +
    `    autotune: ${tuneMs.toFixed(0)}ms loops=${tuneK.loops} lines=${tuneK.lines} quality=${qTune}\n` +
    `    compile overhead: ${speedup}x`
  );
}

describe('autotune stress — compile time & kernel quality at scale', { timeout: 600000 }, () => {

  it('small model: ResNet 5 blocks — autotune baseline', () => {
    const C = 32, S = 8;
    const func = buildFunction('resnet5',
      [tt([1, C, S, S])], [tt([1, C, S, S])],
      (b, args) => {
        let x = args[0];
        for (let i = 0; i < 5; i++) {
          const w = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const g = b.constant(1.0, tt([C])).getResult(0);
          const bt = b.constant(0.0, tt([C])).getResult(0);
          const m = b.constant(0.0, tt([C])).getResult(0);
          const v = b.constant(1.0, tt([C])).getResult(0);
          const conv = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
          const bn = b.batchnorm(conv.getResult(0), g, bt, m, v);
          x = b.relu(b.add(bn.getResult(0), x).getResult(0)).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const ops = countOps(func);

    const t0 = performance.now();
    const rBase = compileBaseline(func);
    const baseMs = performance.now() - t0;

    const func2 = buildFunction('resnet5_t',
      [tt([1, C, S, S])], [tt([1, C, S, S])],
      (b, args) => {
        let x = args[0];
        for (let i = 0; i < 5; i++) {
          const w = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const g = b.constant(1.0, tt([C])).getResult(0);
          const bt = b.constant(0.0, tt([C])).getResult(0);
          const m = b.constant(0.0, tt([C])).getResult(0);
          const v = b.constant(1.0, tt([C])).getResult(0);
          const conv = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
          const bn = b.batchnorm(conv.getResult(0), g, bt, m, v);
          x = b.relu(b.add(bn.getResult(0), x).getResult(0)).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const t1 = performance.now();
    const rTune = compileWithAutotune(func2);
    const tuneMs = performance.now() - t1;

    const baseK = analyzeKernel(rBase.getSource('resnet5'));
    const tuneK = analyzeKernel(rTune.getSource('resnet5_t'));
    log('ResNet 5 blocks', ops, baseMs, tuneMs, baseK, tuneK);

    expect(baseK.issues).toHaveLength(0);
    expect(tuneK.issues).toHaveLength(0);
  });

  it('medium model: Transformer 6L — autotune vs baseline', () => {
    const seq = 8, d = 32;

    function buildTransformer(name) {
      const params = [];
      for (let i = 0; i < 6; i++) {
        params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
        params.push(tt([d]), tt([d]));
        params.push(tt([d, d * 2]), tt([d * 2, d]));
        params.push(tt([d]), tt([d]));
      }
      return buildFunction(name,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          let pi = 1;
          for (let layer = 0; layer < 6; layer++) {
            const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
            const g1 = args[pi++], b1 = args[pi++];
            const wF1 = args[pi++], wF2 = args[pi++];
            const g2 = args[pi++], b2 = args[pi++];

            const ln1 = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g1, b1, 2).getResult(0);
            const ln1r = b.reshape(ln1, [seq, d]).getResult(0);
            const Q = b.matmul(ln1r, wQ).getResult(0);
            const K = b.matmul(ln1r, wK).getResult(0);
            const V = b.matmul(ln1r, wV).getResult(0);
            const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
            const scale = b.broadcast(b.scalarConstant(1 / Math.sqrt(d), F32).getResult(0), [seq, seq], []).getResult(0);
            const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
            const out = b.matmul(attn, V).getResult(0);
            const proj = b.matmul(out, wO).getResult(0);
            x = b.add(x, proj).getResult(0);

            const ln2 = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g2, b2, 2).getResult(0);
            const ln2r = b.reshape(ln2, [seq, d]).getResult(0);
            const ff1 = b.gelu(b.matmul(ln2r, wF1).getResult(0)).getResult(0);
            const ff2 = b.matmul(ff1, wF2).getResult(0);
            x = b.add(x, ff2).getResult(0);
          }
          b.returnOp([x]);
        }
      );
    }

    const funcBase = buildTransformer('tf6_base');
    const funcTune = buildTransformer('tf6_tune');
    const ops = countOps(funcBase);

    const t0 = performance.now();
    const rBase = compileBaseline(funcBase);
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    const rTune = compileWithAutotune(funcTune);
    const tuneMs = performance.now() - t1;

    const baseK = analyzeKernel(rBase.getSource('tf6_base'));
    const tuneK = analyzeKernel(rTune.getSource('tf6_tune'));
    log('Transformer 6L', ops, baseMs, tuneMs, baseK, tuneK);

    expect(baseK.issues).toHaveLength(0);
    expect(tuneK.issues).toHaveLength(0);
  });

  it('large model: GPT-24L (~2k loops) — autotune overhead', () => {
    const seq = 8, d = 32;

    function buildGPT(name, nLayers) {
      const params = [];
      for (let i = 0; i < nLayers; i++) {
        params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
        params.push(tt([d]), tt([d]));
        params.push(tt([d, d * 2]), tt([d * 2, d]));
        params.push(tt([d]), tt([d]));
      }
      return buildFunction(name,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          let pi = 1;
          for (let layer = 0; layer < nLayers; layer++) {
            const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
            const g1 = args[pi++], b1 = args[pi++];
            const wF1 = args[pi++], wF2 = args[pi++];
            const g2 = args[pi++], b2 = args[pi++];

            const ln1 = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g1, b1, 2).getResult(0);
            const ln1r = b.reshape(ln1, [seq, d]).getResult(0);
            const Q = b.matmul(ln1r, wQ).getResult(0);
            const K = b.matmul(ln1r, wK).getResult(0);
            const V = b.matmul(ln1r, wV).getResult(0);
            const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
            const sc = b.broadcast(b.scalarConstant(0.125, F32).getResult(0), [seq, seq], []).getResult(0);
            const a = b.softmax(b.mul(s, sc).getResult(0), 1).getResult(0);
            const o = b.matmul(a, V).getResult(0);
            x = b.add(x, b.matmul(o, wO).getResult(0)).getResult(0);

            const ln2 = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g2, b2, 2).getResult(0);
            const ln2r = b.reshape(ln2, [seq, d]).getResult(0);
            const ff1 = b.gelu(b.matmul(ln2r, wF1).getResult(0)).getResult(0);
            const ff2 = b.matmul(ff1, wF2).getResult(0);
            x = b.add(x, ff2).getResult(0);
          }
          b.returnOp([x]);
        }
      );
    }

    const funcBase = buildGPT('gpt24_base', 24);
    const funcTune = buildGPT('gpt24_tune', 24);
    const ops = countOps(funcBase);

    const t0 = performance.now();
    const rBase = compileBaseline(funcBase);
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    const rTune = compileWithAutotune(funcTune);
    const tuneMs = performance.now() - t1;

    const baseK = analyzeKernel(rBase.getSource('gpt24_base'));
    const tuneK = analyzeKernel(rTune.getSource('gpt24_tune'));
    log('GPT-24L (~2k loops)', ops, baseMs, tuneMs, baseK, tuneK);

    expect(tuneK.issues).toHaveLength(0);
  });

  it('XL model: 50L transformer (~3k loops) — autotune at scale', () => {
    const seq = 4, d = 16;

    function buildXL(name, nLayers) {
      const params = [];
      for (let i = 0; i < nLayers; i++) {
        params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
        params.push(tt([d]), tt([d]));
      }
      return buildFunction(name,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          let pi = 1;
          for (let i = 0; i < nLayers; i++) {
            const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
            const g = args[pi++], bt = args[pi++];
            const Q = b.matmul(x, wQ).getResult(0);
            const K = b.matmul(x, wK).getResult(0);
            const V = b.matmul(x, wV).getResult(0);
            const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
            const sc = b.broadcast(b.scalarConstant(0.25, F32).getResult(0), [seq, seq], []).getResult(0);
            const a = b.softmax(b.mul(s, sc).getResult(0), 1).getResult(0);
            const o = b.matmul(a, V).getResult(0);
            x = b.add(x, b.matmul(o, wO).getResult(0)).getResult(0);
            x = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g, bt, 2).getResult(0);
            x = b.reshape(x, [seq, d]).getResult(0);
          }
          b.returnOp([x]);
        }
      );
    }

    const funcBase = buildXL('xl50_base', 50);
    const funcTune = buildXL('xl50_tune', 50);
    const ops = countOps(funcBase);

    const t0 = performance.now();
    const rBase = compileBaseline(funcBase);
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    const rTune = compileWithAutotune(funcTune);
    const tuneMs = performance.now() - t1;

    const baseK = analyzeKernel(rBase.getSource('xl50_base'));
    const tuneK = analyzeKernel(rTune.getSource('xl50_tune'));
    log('XL 50L (~2800 loops)', ops, baseMs, tuneMs, baseK, tuneK);

    expect(tuneK.issues).toHaveLength(0);
  });

  it('mega model: 100L transformer (~5600 loops) — autotune stress', () => {
    const seq = 4, d = 16;

    function buildMega(name, nLayers) {
      const params = [];
      for (let i = 0; i < nLayers; i++) {
        params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
        params.push(tt([d]), tt([d]));
      }
      return buildFunction(name,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          let pi = 1;
          for (let i = 0; i < nLayers; i++) {
            const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
            const g = args[pi++], bt = args[pi++];
            const Q = b.matmul(x, wQ).getResult(0);
            const K = b.matmul(x, wK).getResult(0);
            const V = b.matmul(x, wV).getResult(0);
            const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
            const sc = b.broadcast(b.scalarConstant(0.25, F32).getResult(0), [seq, seq], []).getResult(0);
            const a = b.softmax(b.mul(s, sc).getResult(0), 1).getResult(0);
            const o = b.matmul(a, V).getResult(0);
            x = b.add(x, b.matmul(o, wO).getResult(0)).getResult(0);
            x = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g, bt, 2).getResult(0);
            x = b.reshape(x, [seq, d]).getResult(0);
          }
          b.returnOp([x]);
        }
      );
    }

    const funcBase = buildMega('mega100_base', 100);
    const funcTune = buildMega('mega100_tune', 100);
    const ops = countOps(funcBase);

    const t0 = performance.now();
    const rBase = compileBaseline(funcBase);
    const baseMs = performance.now() - t0;

    const t1 = performance.now();
    const rTune = compileWithAutotune(funcTune);
    const tuneMs = performance.now() - t1;

    const baseK = analyzeKernel(rBase.getSource('mega100_base'));
    const tuneK = analyzeKernel(rTune.getSource('mega100_tune'));
    log('Mega 100L (~5600 loops)', ops, baseMs, tuneMs, baseK, tuneK);

    expect(tuneK.issues).toHaveLength(0);
  });

  it('monster model: 200L transformer (~11k loops) — autotune at 10k+ ops', () => {
    const seq = 4, d = 16;

    function buildMonster(name, nLayers) {
      const params = [];
      for (let i = 0; i < nLayers; i++) {
        params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
        params.push(tt([d]), tt([d]));
      }
      return buildFunction(name,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          let pi = 1;
          for (let i = 0; i < nLayers; i++) {
            const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
            const g = args[pi++], bt = args[pi++];
            const Q = b.matmul(x, wQ).getResult(0);
            const K = b.matmul(x, wK).getResult(0);
            const V = b.matmul(x, wV).getResult(0);
            const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
            const sc = b.broadcast(b.scalarConstant(0.25, F32).getResult(0), [seq, seq], []).getResult(0);
            const a = b.softmax(b.mul(s, sc).getResult(0), 1).getResult(0);
            const o = b.matmul(a, V).getResult(0);
            x = b.add(x, b.matmul(o, wO).getResult(0)).getResult(0);
            x = b.layernorm(b.reshape(x, [1, seq, d]).getResult(0), g, bt, 2).getResult(0);
            x = b.reshape(x, [seq, d]).getResult(0);
          }
          b.returnOp([x]);
        }
      );
    }

    const funcBase = buildMonster('mon200_base', 200);
    const funcTune = buildMonster('mon200_tune', 200);
    const ops = countOps(funcBase);

    const t0 = performance.now();
    const rBase = compileBaseline(funcBase);
    const baseMs = performance.now() - t0;

    const baseK = analyzeKernel(rBase.getSource('mon200_base'));

    const t1 = performance.now();
    const rTune = compileWithAutotune(funcTune);
    const tuneMs = performance.now() - t1;

    const tuneK = analyzeKernel(rTune.getSource('mon200_tune'));
    log('Monster 200L (~11k loops)', ops, baseMs, tuneMs, baseK, tuneK);

    expect(tuneK.issues).toHaveLength(0);
    expect(tuneK.loops).toBeGreaterThan(5000);
    console.log(`  → ${tuneK.loops} post-autotune loops (baseline ${baseK.loops}) ✓`);
  });

  it('correctness: autotuned kernel produces same output as baseline', () => {
    const seq = 4, d = 8;
    const params = [tt([d, d]), tt([d, d])];

    function buildModel(name) {
      return buildFunction(name,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          const Q = b.matmul(x, args[1]).getResult(0);
          const K = b.matmul(x, args[2]).getResult(0);
          const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const a = b.softmax(s, 1).getResult(0);
          const out = b.matmul(a, x).getResult(0);
          x = b.add(x, out).getResult(0);
          x = b.relu(x).getResult(0);
          b.returnOp([x]);
        }
      );
    }

    const rBase = compileBaseline(buildModel('corr_base'));
    const rTune = compileWithAutotune(buildModel('corr_tune'));

    const inp = new Float32Array(seq * d);
    for (let i = 0; i < inp.length; i++) inp[i] = (i % 7 - 3) * 0.1;
    const w1 = new Float32Array(d * d);
    for (let i = 0; i < w1.length; i++) w1[i] = (i % 5 - 2) * 0.05;
    const w2 = new Float32Array(d * d);
    for (let i = 0; i < w2.length; i++) w2[i] = (i % 3 - 1) * 0.1;

    const outBase = new Float32Array(seq * d);
    const outTune = new Float32Array(seq * d);
    rBase.run('corr_base', inp, w1, w2, outBase);
    rTune.run('corr_tune', inp, w1, w2, outTune);

    for (let i = 0; i < outBase.length; i++) {
      expect(Math.abs(outBase[i] - outTune[i])).toBeLessThan(1e-5);
    }
    console.log('  [Correctness] autotuned output matches baseline ✓');
  });
});
