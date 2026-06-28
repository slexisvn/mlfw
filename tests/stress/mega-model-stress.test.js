import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType, ScalarType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/backend/target.js';

const F32 = ScalarType.F32;
const target = CPUTarget();

function compile(func) {
  return compileGraph(func, target);
}

function countOps(func) {
  let count = 0;
  for (const op of func.ops()) count++;
  return count;
}

function analyzeKernel(src) {
  const loops = (src.match(/\bfor\s*\(/g) || []).length;
  const temps = (src.match(/\bnew Float32Array\(/g) || []).length;
  const lines = src.split('\n').length;
  const hasNegNoise = /\(0\s*-\s*\w/.test(src);
  const hasMulOne = /\*\s*1\b(?!\.)/.test(src);
  const hasAddZero = /\+\s*0(?!\.\d*[1-9])\b/.test(src);
  const issues = [];
  if (hasNegNoise) issues.push('NEG_NOISE');
  if (hasMulOne) issues.push('MUL_ONE');
  if (hasAddZero) issues.push('ADD_ZERO');
  return { loops, temps, lines, issues };
}

function tt(shape) { return new TensorType(shape, F32); }

function log(name, graphOps, compileMs, k) {
  const qStr = k.issues.length === 0 ? 'CLEAN' : k.issues.join(', ');
  console.log(
    `  [${name}] graphOps=${graphOps} loops=${k.loops} temps=${k.temps} ` +
    `lines=${k.lines} compile=${compileMs.toFixed(0)}ms quality=${qStr}`
  );
}

describe('mega model stress — tens of thousands of ops', { timeout: 120000 }, () => {

  it('ResNet-152 scale (75 residual blocks, conv+bn+relu)', () => {
    const C = 32, S = 8;
    const func = buildFunction('resnet152',
      [tt([1, C, S, S])], [tt([1, C, S, S])],
      (b, args) => {
        let x = args[0];
        for (let block = 0; block < 75; block++) {
          const w1 = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const g1 = b.constant(1.0, tt([C])).getResult(0);
          const b1 = b.constant(0.0, tt([C])).getResult(0);
          const m1 = b.constant(0.0, tt([C])).getResult(0);
          const v1 = b.constant(1.0, tt([C])).getResult(0);
          const c1 = b.conv(x, w1, [1, 1], [[1, 1], [1, 1]]);
          const bn1 = b.batchnorm(c1.getResult(0), g1, b1, m1, v1);
          const r1 = b.relu(bn1.getResult(0));
          const w2 = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const g2 = b.constant(1.0, tt([C])).getResult(0);
          const b2 = b.constant(0.0, tt([C])).getResult(0);
          const m2 = b.constant(0.0, tt([C])).getResult(0);
          const v2 = b.constant(1.0, tt([C])).getResult(0);
          const c2 = b.conv(r1.getResult(0), w2, [1, 1], [[1, 1], [1, 1]]);
          const bn2 = b.batchnorm(c2.getResult(0), g2, b2, m2, v2);
          x = b.relu(b.add(bn2.getResult(0), x).getResult(0)).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);
    expect(graphOps).toBeGreaterThan(1500);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('resnet152');
    const k = analyzeKernel(src);
    log('ResNet-152 (75 blocks)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(1000);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('GPT-XL (24 transformer layers, attention+FFN+LN+GELU)', () => {
    const seq = 16, d = 64, dFF = 128;

    const params = [];
    for (let i = 0; i < 24; i++) {
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
      params.push(tt([d, dFF]), tt([dFF, d]));
      params.push(tt([d]), tt([d]));
    }

    const func = buildFunction('gpt_xl',
      [tt([seq, d]), ...params], [tt([seq, d])],
      (b, args) => {
        let x = args[0];
        let pi = 1;
        for (let layer = 0; layer < 24; layer++) {
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

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('gpt_xl');
    const k = analyzeKernel(src);
    log('GPT-XL (24 layers)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(2000);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('deep MobileNet (50 inverted residual blocks, expand+DW+SE+project)', () => {
    const C = 16, S = 8;

    const func = buildFunction('deep_mobilenet',
      [tt([1, C, S, S])], [tt([1, C, S, S])],
      (b, args) => {
        let x = args[0];
        const expand = C * 4;

        for (let block = 0; block < 50; block++) {
          const wE = b.constant(0.01, tt([expand, C, 1, 1])).getResult(0);
          const ex = b.silu(b.conv(x, wE, [1, 1], [[0, 0], [0, 0]]).getResult(0));

          const wDW = b.constant(0.01, tt([expand, 1, 3, 3])).getResult(0);
          const dw = b.silu(b.conv(ex.getResult(0), wDW, [1, 1], [[1, 1], [1, 1]], { groups: expand }).getResult(0));

          const seC = Math.max(1, C / 4);
          const wSE1 = b.constant(0.01, tt([seC, expand, 1, 1])).getResult(0);
          const wSE2 = b.constant(0.01, tt([expand, seC, 1, 1])).getResult(0);
          const pooled = b.reduce(dw.getResult(0), b.scalarConstant(0, F32).getResult(0), [2, 3], 'mean');
          const poolR = b.reshape(pooled.getResult(0), [1, expand, 1, 1]).getResult(0);
          const se1 = b.silu(b.conv(poolR, wSE1, [1, 1], [[0, 0], [0, 0]]).getResult(0));
          const se2 = b.sigmoid(b.conv(se1.getResult(0), wSE2, [1, 1], [[0, 0], [0, 0]]).getResult(0));
          const seBcast = b.broadcast(
            b.reshape(se2.getResult(0), [1, expand, 1, 1]).getResult(0),
            [1, expand, S, S], [0, 1, 2, 3]
          ).getResult(0);
          const scaled = b.mul(dw.getResult(0), seBcast);

          const wP = b.constant(0.01, tt([C, expand, 1, 1])).getResult(0);
          const proj = b.conv(scaled.getResult(0), wP, [1, 1], [[0, 0], [0, 0]]);

          const g = b.constant(1.0, tt([C])).getResult(0);
          const bt = b.constant(0.0, tt([C])).getResult(0);
          const m = b.constant(0.0, tt([C])).getResult(0);
          const v = b.constant(1.0, tt([C])).getResult(0);
          const bn = b.batchnorm(proj.getResult(0), g, bt, m, v);
          x = b.add(bn.getResult(0), x).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('deep_mobilenet');
    const k = analyzeKernel(src);
    log('Deep MobileNet (50 blocks+SE)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(1500);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('BERT-Large (12 layers, multi-head attn + FFN + LN + GELU)', () => {
    const seq = 16, d = 64, dFF = 256, nHeads = 8, dHead = d / nHeads;

    const params = [];
    for (let i = 0; i < 12; i++) {
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
      params.push(tt([d, dFF]), tt([dFF, d]));
      params.push(tt([d]), tt([d]));
    }

    const func = buildFunction('bert_large',
      [tt([seq, d]), ...params], [tt([seq, d])],
      (b, args) => {
        let x = args[0];
        let pi = 1;
        for (let layer = 0; layer < 12; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], b1 = args[pi++];
          const wF1 = args[pi++], wF2 = args[pi++];
          const g2 = args[pi++], b2 = args[pi++];

          const Q = b.matmul(x, wQ).getResult(0);
          const K = b.matmul(x, wK).getResult(0);
          const V = b.matmul(x, wV).getResult(0);
          const Qh = b.reshape(Q, [seq, nHeads, dHead]).getResult(0);
          const Kh = b.reshape(K, [seq, nHeads, dHead]).getResult(0);
          const Vh = b.reshape(V, [seq, nHeads, dHead]).getResult(0);
          const Qt = b.transpose(Qh, [1, 0, 2]).getResult(0);
          const Kt = b.transpose(Kh, [1, 2, 0]).getResult(0);
          const Vt = b.transpose(Vh, [1, 0, 2]).getResult(0);
          const scores = b.matmul(Qt, Kt).getResult(0);
          const scale = b.broadcast(b.scalarConstant(1 / Math.sqrt(dHead), F32).getResult(0), [nHeads, seq, seq], []).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 2).getResult(0);
          const attnOut = b.matmul(attn, Vt).getResult(0);
          const attnT = b.transpose(attnOut, [1, 0, 2]).getResult(0);
          const flat = b.reshape(attnT, [seq, d]).getResult(0);
          const proj = b.matmul(flat, wO).getResult(0);
          const res1 = b.add(x, proj).getResult(0);
          const ln1 = b.layernorm(b.reshape(res1, [1, seq, d]).getResult(0), g1, b1, 2).getResult(0);
          const ln1r = b.reshape(ln1, [seq, d]).getResult(0);

          const ff1 = b.gelu(b.matmul(ln1r, wF1).getResult(0)).getResult(0);
          const ff2 = b.matmul(ff1, wF2).getResult(0);
          const res2 = b.add(ln1r, ff2).getResult(0);
          x = b.layernorm(b.reshape(res2, [1, seq, d]).getResult(0), g2, b2, 2).getResult(0);
          x = b.reshape(x, [seq, d]).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('bert_large');
    const k = analyzeKernel(src);
    log('BERT-Large (12 layers, 8 heads)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(1200);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('mega DenseNet (30 dense layers, each concats all previous)', () => {
    const growthRate = 4, initC = 8, S = 4, nLayers = 30;
    const totalC = initC + growthRate * nLayers;

    const func = buildFunction('mega_densenet',
      [tt([1, initC, S, S])], [tt([1, totalC, S, S])],
      (b, args) => {
        const features = [args[0]];
        let curC = initC;

        for (let i = 0; i < nLayers; i++) {
          const x = features.length === 1 ? features[0] : b.concat(features, 1).getResult(0);

          const wBN = b.constant(0.01, tt([growthRate * 4, curC, 1, 1])).getResult(0);
          const g1 = b.constant(1.0, tt([growthRate * 4])).getResult(0);
          const bt1 = b.constant(0.0, tt([growthRate * 4])).getResult(0);
          const m1 = b.constant(0.0, tt([growthRate * 4])).getResult(0);
          const v1 = b.constant(1.0, tt([growthRate * 4])).getResult(0);
          const bn1 = b.batchnorm(b.conv(x, wBN, [1, 1], [[0, 0], [0, 0]]).getResult(0), g1, bt1, m1, v1);
          const act1 = b.relu(bn1.getResult(0));

          const w3 = b.constant(0.01, tt([growthRate, growthRate * 4, 3, 3])).getResult(0);
          const g2 = b.constant(1.0, tt([growthRate])).getResult(0);
          const bt2 = b.constant(0.0, tt([growthRate])).getResult(0);
          const m2 = b.constant(0.0, tt([growthRate])).getResult(0);
          const v2 = b.constant(1.0, tt([growthRate])).getResult(0);
          const bn2 = b.batchnorm(b.conv(act1.getResult(0), w3, [1, 1], [[1, 1], [1, 1]]).getResult(0), g2, bt2, m2, v2);
          const act2 = b.relu(bn2.getResult(0));

          features.push(act2.getResult(0));
          curC += growthRate;
        }

        const out = b.concat(features, 1);
        b.returnOp([out.getResult(0)]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('mega_densenet');
    const k = analyzeKernel(src);
    log('Mega DenseNet (30 dense layers)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(1000);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('Seq2Seq-XL (8 enc + 8 dec with cross-attn + LN + GELU FFN)', () => {
    const seq = 8, d = 32, dFF = 64, nLayers = 8;

    const params = [];
    for (let i = 0; i < nLayers; i++) {
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
      params.push(tt([d, dFF]), tt([dFF, d]));
      params.push(tt([d]), tt([d]));
    }
    for (let i = 0; i < nLayers; i++) {
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
      params.push(tt([d, dFF]), tt([dFF, d]));
      params.push(tt([d]), tt([d]));
    }

    const func = buildFunction('seq2seq_xl',
      [tt([seq, d]), tt([seq, d]), ...params], [tt([seq, d])],
      (b, args) => {
        let enc = args[0], dec = args[1];
        let pi = 2;

        for (let layer = 0; layer < nLayers; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], b1 = args[pi++];
          const wF1 = args[pi++], wF2 = args[pi++];
          const g2 = args[pi++], b2 = args[pi++];

          const Q = b.matmul(enc, wQ).getResult(0);
          const K = b.matmul(enc, wK).getResult(0);
          const V = b.matmul(enc, wV).getResult(0);
          const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const sc = b.broadcast(b.scalarConstant(1 / Math.sqrt(d), F32).getResult(0), [seq, seq], []).getResult(0);
          const a = b.softmax(b.mul(s, sc).getResult(0), 1).getResult(0);
          const o = b.matmul(a, V).getResult(0);
          const p = b.matmul(o, wO).getResult(0);
          const r1 = b.add(enc, p).getResult(0);
          const ln1 = b.layernorm(b.reshape(r1, [1, seq, d]).getResult(0), g1, b1, 2).getResult(0);

          const ln1r = b.reshape(ln1, [seq, d]).getResult(0);
          const f1 = b.gelu(b.matmul(ln1r, wF1).getResult(0)).getResult(0);
          const f2 = b.matmul(f1, wF2).getResult(0);
          const r2 = b.add(ln1r, f2).getResult(0);
          enc = b.layernorm(b.reshape(r2, [1, seq, d]).getResult(0), g2, b2, 2).getResult(0);
          enc = b.reshape(enc, [seq, d]).getResult(0);
        }

        for (let layer = 0; layer < nLayers; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], b1 = args[pi++];
          const cwQ = args[pi++], cwK = args[pi++], cwV = args[pi++], cwO = args[pi++];
          const cg = args[pi++], cb = args[pi++];
          const wF1 = args[pi++], wF2 = args[pi++];
          const g2 = args[pi++], b2 = args[pi++];

          const Q = b.matmul(dec, wQ).getResult(0);
          const K = b.matmul(dec, wK).getResult(0);
          const V = b.matmul(dec, wV).getResult(0);
          const s = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const sc = b.broadcast(b.scalarConstant(1 / Math.sqrt(d), F32).getResult(0), [seq, seq], []).getResult(0);
          const a = b.softmax(b.mul(s, sc).getResult(0), 1).getResult(0);
          const o = b.matmul(a, V).getResult(0);
          const p = b.matmul(o, wO).getResult(0);
          const r1 = b.add(dec, p).getResult(0);
          const ln1 = b.layernorm(b.reshape(r1, [1, seq, d]).getResult(0), g1, b1, 2).getResult(0);
          const ln1r = b.reshape(ln1, [seq, d]).getResult(0);

          const cQ = b.matmul(ln1r, cwQ).getResult(0);
          const cK = b.matmul(enc, cwK).getResult(0);
          const cV = b.matmul(enc, cwV).getResult(0);
          const cs = b.matmul(cQ, b.transpose(cK, [1, 0]).getResult(0)).getResult(0);
          const csc = b.broadcast(b.scalarConstant(1 / Math.sqrt(d), F32).getResult(0), [seq, seq], []).getResult(0);
          const ca = b.softmax(b.mul(cs, csc).getResult(0), 1).getResult(0);
          const co = b.matmul(ca, cV).getResult(0);
          const cp = b.matmul(co, cwO).getResult(0);
          const cr = b.add(ln1r, cp).getResult(0);
          const cln = b.layernorm(b.reshape(cr, [1, seq, d]).getResult(0), cg, cb, 2).getResult(0);
          const clnr = b.reshape(cln, [seq, d]).getResult(0);

          const f1 = b.gelu(b.matmul(clnr, wF1).getResult(0)).getResult(0);
          const f2 = b.matmul(f1, wF2).getResult(0);
          const r2 = b.add(clnr, f2).getResult(0);
          dec = b.layernorm(b.reshape(r2, [1, seq, d]).getResult(0), g2, b2, 2).getResult(0);
          dec = b.reshape(dec, [seq, d]).getResult(0);
        }
        b.returnOp([dec]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('seq2seq_xl');
    const k = analyzeKernel(src);
    log('Seq2Seq-XL (8+8 layers)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(1800);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(120000);
  });

  it('deep MLP-Mixer (60 layers, token mixing + channel mixing + LN)', () => {
    const tokens = 16, channels = 32;

    const params = [];
    for (let i = 0; i < 60; i++) {
      params.push(tt([channels]));
      params.push(tt([channels]));
      params.push(tt([tokens, tokens]));
      params.push(tt([tokens, tokens]));
      params.push(tt([channels]));
      params.push(tt([channels]));
      params.push(tt([channels, channels]));
      params.push(tt([channels, channels]));
    }

    const func = buildFunction('mlp_mixer',
      [tt([tokens, channels]), ...params], [tt([tokens, channels])],
      (b, args) => {
        let x = args[0];
        let pi = 1;

        for (let layer = 0; layer < 60; layer++) {
          const g1 = args[pi++], b1 = args[pi++];
          const wT1 = args[pi++], wT2 = args[pi++];
          const g2 = args[pi++], b2 = args[pi++];
          const wC1 = args[pi++], wC2 = args[pi++];

          const ln1 = b.layernorm(b.reshape(x, [1, tokens, channels]).getResult(0), g1, b1, 2).getResult(0);
          const ln1r = b.reshape(ln1, [tokens, channels]).getResult(0);
          const xT = b.transpose(ln1r, [1, 0]).getResult(0);
          const t1 = b.gelu(b.matmul(xT, wT1).getResult(0)).getResult(0);
          const t2 = b.matmul(t1, wT2).getResult(0);
          const tOut = b.transpose(t2, [1, 0]).getResult(0);
          x = b.add(x, tOut).getResult(0);

          const ln2 = b.layernorm(b.reshape(x, [1, tokens, channels]).getResult(0), g2, b2, 2).getResult(0);
          const ln2r = b.reshape(ln2, [tokens, channels]).getResult(0);
          const c1 = b.gelu(b.matmul(ln2r, wC1).getResult(0)).getResult(0);
          const c2 = b.matmul(c1, wC2).getResult(0);
          x = b.add(x, c2).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('mlp_mixer');
    const k = analyzeKernel(src);
    log('MLP-Mixer (60 layers)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(3000);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(120000);
  });

  it('mega multi-branch: ResNet + Transformer + MLP fused pipeline', () => {
    const C = 16, S = 8, d = 32, seq = 16;

    const params = [];
    for (let i = 0; i < 8; i++) {
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
      params.push(tt([d, d * 2]), tt([d * 2, d]));
      params.push(tt([d]), tt([d]));
    }

    const func = buildFunction('mega_combined',
      [tt([1, C, S, S]), tt([seq, d]), ...params],
      [tt([1, C, S, S]), tt([seq, d])],
      (b, args) => {
        let cnn = args[0];
        let seq_x = args[1];
        let pi = 2;

        for (let block = 0; block < 30; block++) {
          const w1 = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const g1 = b.constant(1.0, tt([C])).getResult(0);
          const bt1 = b.constant(0.0, tt([C])).getResult(0);
          const m1 = b.constant(0.0, tt([C])).getResult(0);
          const v1 = b.constant(1.0, tt([C])).getResult(0);
          const conv = b.conv(cnn, w1, [1, 1], [[1, 1], [1, 1]]);
          const bn = b.batchnorm(conv.getResult(0), g1, bt1, m1, v1);
          cnn = b.relu(b.add(bn.getResult(0), cnn).getResult(0)).getResult(0);
        }

        for (let layer = 0; layer < 8; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], b1 = args[pi++];
          const wF1 = args[pi++], wF2 = args[pi++];
          const g2 = args[pi++], b2 = args[pi++];

          const ln1 = b.layernorm(b.reshape(seq_x, [1, seq, d]).getResult(0), g1, b1, 2).getResult(0);
          const ln1r = b.reshape(ln1, [seq, d]).getResult(0);
          const Q = b.matmul(ln1r, wQ).getResult(0);
          const K = b.matmul(ln1r, wK).getResult(0);
          const V = b.matmul(ln1r, wV).getResult(0);
          const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale = b.broadcast(b.scalarConstant(1 / Math.sqrt(d), F32).getResult(0), [seq, seq], []).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
          const out = b.matmul(attn, V).getResult(0);
          const proj = b.matmul(out, wO).getResult(0);
          seq_x = b.add(seq_x, proj).getResult(0);

          const ln2 = b.layernorm(b.reshape(seq_x, [1, seq, d]).getResult(0), g2, b2, 2).getResult(0);
          const ln2r = b.reshape(ln2, [seq, d]).getResult(0);
          const ff1 = b.gelu(b.matmul(ln2r, wF1).getResult(0)).getResult(0);
          const ff2 = b.matmul(ff1, wF2).getResult(0);
          seq_x = b.add(seq_x, ff2).getResult(0);
        }

        b.returnOp([cnn, seq_x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('mega_combined');
    const k = analyzeKernel(src);
    log('Mega Combined (30 ResBlk + 8 TF)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(900);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('Inception-ResNet (20 modules, 4 branches each + BN + residual)', () => {
    const C = 16, S = 8;

    const func = buildFunction('inception_resnet',
      [tt([1, C * 4, S, S])], [tt([1, C * 4, S, S])],
      (b, args) => {
        let x = args[0];
        const inC = C * 4;

        for (let mod = 0; mod < 20; mod++) {
          const w1x1 = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const b1 = b.relu(b.conv(x, w1x1, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);

          const w3r = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const r3 = b.relu(b.conv(x, w3r, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);
          const w3 = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const b3 = b.relu(b.conv(r3, w3, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);

          const w5r = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const r5 = b.relu(b.conv(x, w5r, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);
          const w5a = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const c5a = b.relu(b.conv(r5, w5a, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);
          const w5b = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const b5 = b.relu(b.conv(c5a, w5b, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);

          const wp = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const bp = b.relu(b.conv(x, wp, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);

          const cat = b.concat([b1, b3, b5, bp], 1).getResult(0);

          const gBN = b.constant(1.0, tt([inC])).getResult(0);
          const btBN = b.constant(0.0, tt([inC])).getResult(0);
          const mBN = b.constant(0.0, tt([inC])).getResult(0);
          const vBN = b.constant(1.0, tt([inC])).getResult(0);
          const bn = b.batchnorm(cat, gBN, btBN, mBN, vBN);

          x = b.relu(b.add(bn.getResult(0), x).getResult(0)).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('inception_resnet');
    const k = analyzeKernel(src);
    log('Inception-ResNet (20 modules)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(500);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(60000);
  });

  it('ultra-deep residual MLP (100 layers, alternating gelu/silu/sigmoid + LN)', () => {
    const d = 32;

    const params = [];
    for (let i = 0; i < 100; i++) {
      params.push(tt([d, d]));
      params.push(tt([d]));
      params.push(tt([d]));
    }

    const func = buildFunction('ultra_mlp',
      [tt([1, d]), ...params], [tt([1, d])],
      (b, args) => {
        let x = args[0];
        let pi = 1;
        for (let i = 0; i < 100; i++) {
          const w = args[pi++], g = args[pi++], bt = args[pi++];
          const h = b.matmul(x, w).getResult(0);

          let act;
          if (i % 4 === 0) act = b.gelu(h).getResult(0);
          else if (i % 4 === 1) act = b.silu(h).getResult(0);
          else if (i % 4 === 2) act = b.sigmoid(h).getResult(0);
          else act = b.tanh(h).getResult(0);

          const normed = b.layernorm(b.reshape(act, [1, 1, d]).getResult(0), g, bt, 2).getResult(0);
          x = b.add(x, b.reshape(normed, [1, d]).getResult(0)).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('ultra_mlp');
    const k = analyzeKernel(src);
    log('Ultra MLP (100 layers)', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(1000);
    expect(k.issues).toHaveLength(0);
    expect(ms).toBeLessThan(120000);
  });

  it('10k+ ops: monster 200-layer transformer → compile and verify kernel quality', () => {
    const seq = 4, d = 16;

    const params = [];
    for (let i = 0; i < 200; i++) {
      params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
      params.push(tt([d]), tt([d]));
    }

    const func = buildFunction('monster_200L',
      [tt([seq, d]), ...params], [tt([seq, d])],
      (b, args) => {
        let x = args[0];
        let pi = 1;
        for (let i = 0; i < 200; i++) {
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

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('monster_200L');
    const k = analyzeKernel(src);
    log('Monster 200L Transformer', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(10000);
    expect(k.issues).toHaveLength(0);
    console.log(`  → ${k.loops} loops = ${k.loops} post-decomp ops (tens of thousands ✓)`);
  });

  it('10k+ ops: 150-layer MLP-Mixer (token + channel mixing + LN + GELU)', () => {
    const tokens = 8, channels = 16;

    const params = [];
    for (let i = 0; i < 150; i++) {
      params.push(tt([channels]), tt([channels]));
      params.push(tt([tokens, tokens]), tt([tokens, tokens]));
      params.push(tt([channels]), tt([channels]));
      params.push(tt([channels, channels]), tt([channels, channels]));
    }

    const func = buildFunction('mega_mixer_150',
      [tt([tokens, channels]), ...params], [tt([tokens, channels])],
      (b, args) => {
        let x = args[0];
        let pi = 1;
        for (let layer = 0; layer < 150; layer++) {
          const g1 = args[pi++], b1 = args[pi++];
          const wT1 = args[pi++], wT2 = args[pi++];
          const g2 = args[pi++], b2 = args[pi++];
          const wC1 = args[pi++], wC2 = args[pi++];

          const ln1 = b.layernorm(b.reshape(x, [1, tokens, channels]).getResult(0), g1, b1, 2).getResult(0);
          const ln1r = b.reshape(ln1, [tokens, channels]).getResult(0);
          const xT = b.transpose(ln1r, [1, 0]).getResult(0);
          const t1 = b.gelu(b.matmul(xT, wT1).getResult(0)).getResult(0);
          const t2 = b.matmul(t1, wT2).getResult(0);
          const tOut = b.transpose(t2, [1, 0]).getResult(0);
          x = b.add(x, tOut).getResult(0);

          const ln2 = b.layernorm(b.reshape(x, [1, tokens, channels]).getResult(0), g2, b2, 2).getResult(0);
          const ln2r = b.reshape(ln2, [tokens, channels]).getResult(0);
          const c1 = b.gelu(b.matmul(ln2r, wC1).getResult(0)).getResult(0);
          const c2 = b.matmul(c1, wC2).getResult(0);
          x = b.add(x, c2).getResult(0);
        }
        b.returnOp([x]);
      }
    );

    const graphOps = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const ms = performance.now() - t0;

    const src = r.getSource('mega_mixer_150');
    const k = analyzeKernel(src);
    log('Mega Mixer 150L', graphOps, ms, k);

    expect(k.loops).toBeGreaterThan(9000);
    expect(k.issues).toHaveLength(0);
    console.log(`  → ${k.loops} loops = ${k.loops} post-decomp ops (tens of thousands ✓)`);
  });

  it('compile scaling profile across depths', () => {
    const results = [];

    for (const depth of [10, 50, 100, 200]) {
      const d = 16, seq = 4;
      const params = [];
      for (let i = 0; i < depth; i++) {
        params.push(tt([d, d]), tt([d, d]), tt([d, d]), tt([d, d]));
        params.push(tt([d]), tt([d]));
      }

      const func = buildFunction(`scale_${depth}`,
        [tt([seq, d]), ...params], [tt([seq, d])],
        (b, args) => {
          let x = args[0];
          let pi = 1;
          for (let i = 0; i < depth; i++) {
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

      const ops = countOps(func);
      const t0 = performance.now();
      compile(func);
      const ms = performance.now() - t0;
      results.push({ depth, ops, ms });
    }

    for (const r of results) {
      console.log(`  [Scale ${r.depth}L] graphOps=${r.ops} compile=${r.ms.toFixed(0)}ms`);
    }

    expect(results[3].ms).toBeLessThan(120000);
  });
});
