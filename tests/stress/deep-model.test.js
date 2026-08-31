import { describe, it, expect } from 'vitest';
import { buildFunction } from '../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../src/compiler/ir/graph/types.js';
import { compileGraph } from '../../src/compiler/pipeline/compiler.js';
import { CPUTarget } from '../../src/compiler/support/target.js';
import { F32 } from '../_utils/ir_fixture.js';

const target = CPUTarget();

function compile(func, opts = {}) {
  return compileGraph(func, target, opts);
}

function countOps(func) {
  let count = 0;
  for (const op of func.ops()) count++;
  return count;
}

function analyzeKernel(src) {
  const loops = (src.match(/\bfor\s*\(/g) || []).length;
  const temps = (src.match(/\bnew Float32Array\(/g) || []).length;
  const mathCalls = (src.match(/Math\.\w+/g) || []).length;
  const lines = src.split('\n').length;
  const hasNegNoise = /\(0\s*-\s*\w/.test(src);
  const hasMulOne = /\*\s*1\b(?!\.)/.test(src);
  return { loops, temps, mathCalls, lines, hasNegNoise, hasMulOne };
}

function report(name, opCount, compileMs, kernelStats) {
  const quality = [];
  if (kernelStats.hasNegNoise) quality.push('NEG_NOISE');
  if (kernelStats.hasMulOne) quality.push('MUL_ONE');
  const qStr = quality.length === 0 ? 'CLEAN' : quality.join(', ');
  console.log(
    `  [${name}] ops=${opCount} compile=${compileMs.toFixed(0)}ms ` +
    `loops=${kernelStats.loops} temps=${kernelStats.temps} lines=${kernelStats.lines} ` +
    `quality=${qStr}`
  );
  return { quality, compileMs, ...kernelStats };
}

function tt(shape) { return new TensorType(shape, F32); }

describe('deep model stress — compile time & kernel quality', () => {

  it('deep ResNet-50 backbone (20 residual blocks, conv+bn+relu)', () => {
    const input = tt([1, 64, 16, 16]);
    const func = buildFunction('resnet50', [input], [input], (b, args) => {
      let x = args[0];
      const channels = 64;
      for (let block = 0; block < 20; block++) {
        const w1 = b.constant(0.01, tt([channels, channels, 3, 3])).getResult(0);
        const g1 = b.constant(1.0, tt([channels])).getResult(0);
        const bt1 = b.constant(0.0, tt([channels])).getResult(0);
        const m1 = b.constant(0.0, tt([channels])).getResult(0);
        const v1 = b.constant(1.0, tt([channels])).getResult(0);
        const c1 = b.conv(x, w1, [1, 1], [[1, 1], [1, 1]]);
        const bn1 = b.batchnorm(c1.getResult(0), g1, bt1, m1, v1);
        const r1 = b.relu(bn1.getResult(0));
        const w2 = b.constant(0.01, tt([channels, channels, 3, 3])).getResult(0);
        const g2 = b.constant(1.0, tt([channels])).getResult(0);
        const bt2 = b.constant(0.0, tt([channels])).getResult(0);
        const m2 = b.constant(0.0, tt([channels])).getResult(0);
        const v2 = b.constant(1.0, tt([channels])).getResult(0);
        const c2 = b.conv(r1.getResult(0), w2, [1, 1], [[1, 1], [1, 1]]);
        const bn2 = b.batchnorm(c2.getResult(0), g2, bt2, m2, v2);
        const residual = b.add(bn2.getResult(0), x);
        x = b.relu(residual.getResult(0)).getResult(0);
      }
      b.returnOp([x]);
    });

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(400);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('resnet50');
    const stats = analyzeKernel(src);
    const info = report('ResNet-50 (20 blocks)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);

    const inp = new Float32Array(1 * 64 * 16 * 16);
    const out = new Float32Array(1 * 64 * 16 * 16);
    r.run('resnet50', inp, out);
    expect(out.every(v => isFinite(v))).toBe(true);
  });

  it('6-layer Transformer encoder (attention + FFN + layernorm)', () => {
    const seqLen = 8, dModel = 32, dFF = 64, nHeads = 4;
    const dHead = dModel / nHeads;
    const input = tt([1, seqLen, dModel]);

    const paramTypes = [];
    for (let layer = 0; layer < 6; layer++) {
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel, dFF]));
      paramTypes.push(tt([dFF, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
    }

    const func = buildFunction('transformer_enc', [input, ...paramTypes], [input], (b, args) => {
      let x = args[0];
      let pi = 1;

      for (let layer = 0; layer < 6; layer++) {
        const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
        const lnG1 = args[pi++], lnB1 = args[pi++];
        const wFF1 = args[pi++], wFF2 = args[pi++];
        const lnG2 = args[pi++], lnB2 = args[pi++];

        const xr = b.reshape(x, [seqLen, dModel]).getResult(0);
        const Q = b.matmul(xr, wQ).getResult(0);
        const K = b.matmul(xr, wK).getResult(0);
        const V = b.matmul(xr, wV).getResult(0);

        const Qh = b.reshape(Q, [seqLen, nHeads, dHead]).getResult(0);
        const Kh = b.reshape(K, [seqLen, nHeads, dHead]).getResult(0);
        const Vh = b.reshape(V, [seqLen, nHeads, dHead]).getResult(0);

        const Qt = b.transpose(Qh, [1, 0, 2]).getResult(0);
        const Kt = b.transpose(Kh, [1, 2, 0]).getResult(0);
        const Vt = b.transpose(Vh, [1, 0, 2]).getResult(0);

        const scores = b.matmul(Qt, Kt).getResult(0);

        const scaleVal = b.broadcast(
          b.scalarConstant(1.0 / Math.sqrt(dHead), F32).getResult(0),
          [nHeads, seqLen, seqLen], []
        ).getResult(0);
        const scaled = b.mul(scores, scaleVal).getResult(0);
        const attnWeights = b.softmax(scaled, 2).getResult(0);
        const attnOut = b.matmul(attnWeights, Vt).getResult(0);

        const attnT = b.transpose(attnOut, [1, 0, 2]).getResult(0);
        const attnFlat = b.reshape(attnT, [seqLen, dModel]).getResult(0);
        const projected = b.matmul(attnFlat, wO).getResult(0);

        const projR = b.reshape(projected, [1, seqLen, dModel]).getResult(0);
        const res1 = b.add(x, projR).getResult(0);
        const ln1 = b.layernorm(res1, lnG1, lnB1, 2).getResult(0);

        const ln1r = b.reshape(ln1, [seqLen, dModel]).getResult(0);
        const ff1 = b.matmul(ln1r, wFF1).getResult(0);
        const ff1act = b.gelu(ff1).getResult(0);
        const ff2 = b.matmul(ff1act, wFF2).getResult(0);
        const ff2r = b.reshape(ff2, [1, seqLen, dModel]).getResult(0);

        const res2 = b.add(ln1, ff2r).getResult(0);
        x = b.layernorm(res2, lnG2, lnB2, 2).getResult(0);
      }

      b.returnOp([x]);
    });

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(150);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('transformer_enc');
    const stats = analyzeKernel(src);
    const info = report('Transformer 6L', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('U-Net 4-level encoder-decoder with skip connections', () => {
    const func = buildFunction('unet',
      [tt([1, 1, 32, 32])],
      [tt([1, 1, 32, 32])],
      (b, args) => {
        let x = args[0];
        const skips = [];

        const channelSizes = [1, 8, 16, 32];
        const spatialSizes = [32, 16, 8, 4];

        for (let level = 0; level < 3; level++) {
          const cIn = channelSizes[level], cOut = channelSizes[level + 1];
          const w1 = b.constant(0.01, tt([cOut, cIn, 3, 3])).getResult(0);
          const conv1 = b.conv(x, w1, [1, 1], [[1, 1], [1, 1]]);
          const act1 = b.relu(conv1.getResult(0));

          const w2 = b.constant(0.01, tt([cOut, cOut, 3, 3])).getResult(0);
          const conv2 = b.conv(act1.getResult(0), w2, [1, 1], [[1, 1], [1, 1]]);
          const act2 = b.relu(conv2.getResult(0));
          skips.push(act2.getResult(0));

          const wPool = b.constant(0.01, tt([cOut, cOut, 2, 2])).getResult(0);
          const pool = b.conv(act2.getResult(0), wPool, [2, 2], [[0, 0], [0, 0]]);
          x = pool.getResult(0);
        }

        const wBot1 = b.constant(0.01, tt([64, 32, 3, 3])).getResult(0);
        const bot1 = b.relu(b.conv(x, wBot1, [1, 1], [[1, 1], [1, 1]]).getResult(0));
        const wBot2 = b.constant(0.01, tt([32, 64, 3, 3])).getResult(0);
        x = b.relu(b.conv(bot1.getResult(0), wBot2, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);

        for (let level = 2; level >= 0; level--) {
          const sIn = spatialSizes[level + 1];
          const sOut = spatialSizes[level];
          const cSkip = channelSizes[level + 1];
          const cOut = channelSizes[level] === 1 ? 1 : channelSizes[level + 1];
          const cIn = channelSizes[level + 1];

          const xUp = b.resize(x, [sOut, sOut], 'nearest', { layout: 'NCHW' });

          const skip = skips[level];
          const cat = b.concat([xUp.getResult(0), skip], 1);

          const wD1 = b.constant(0.01, tt([cOut, cIn + cSkip, 3, 3])).getResult(0);
          const d1 = b.relu(b.conv(cat.getResult(0), wD1, [1, 1], [[1, 1], [1, 1]]).getResult(0));
          const wD2 = b.constant(0.01, tt([cOut, cOut, 3, 3])).getResult(0);
          x = b.relu(b.conv(d1.getResult(0), wD2, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);
        }

        const wFinal = b.constant(0.01, tt([1, 1, 1, 1])).getResult(0);
        const final = b.conv(x, wFinal, [1, 1], [[0, 0], [0, 0]]);
        b.returnOp([final.getResult(0)]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(50);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('unet');
    const stats = analyzeKernel(src);
    const info = report('U-Net 4-level', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);

    const inp = new Float32Array(1 * 1 * 32 * 32).fill(0.5);
    const out = new Float32Array(1 * 1 * 32 * 32);
    r.run('unet', inp, out);
    expect(out.every(v => isFinite(v))).toBe(true);
  });

  it('DenseNet-style dense block (12 layers, each sees all previous)', () => {
    const growthRate = 4;
    const initChannels = 8;
    const nLayers = 12;
    const spatial = 8;
    const totalChannels = initChannels + growthRate * nLayers;

    const func = buildFunction('densenet',
      [tt([1, initChannels, spatial, spatial])],
      [tt([1, totalChannels, spatial, spatial])],
      (b, args) => {
        const features = [args[0]];
        let currentChannels = initChannels;

        for (let i = 0; i < nLayers; i++) {
          let x;
          if (features.length === 1) {
            x = features[0];
          } else {
            x = b.concat(features, 1).getResult(0);
          }

          const w = b.constant(0.01, tt([growthRate, currentChannels, 3, 3])).getResult(0);
          const g = b.constant(1.0, tt([growthRate])).getResult(0);
          const bt = b.constant(0.0, tt([growthRate])).getResult(0);
          const m = b.constant(0.0, tt([growthRate])).getResult(0);
          const v = b.constant(1.0, tt([growthRate])).getResult(0);

          const conv = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
          const bn = b.batchnorm(conv.getResult(0), g, bt, m, v);
          const act = b.relu(bn.getResult(0));
          features.push(act.getResult(0));
          currentChannels += growthRate;
        }

        const out = b.concat(features, 1);
        b.returnOp([out.getResult(0)]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(100);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('densenet');
    const stats = analyzeKernel(src);
    const info = report('DenseNet (12 layers)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('deep MLP with 30 hidden layers + activations + layernorm', () => {
    const hidden = 64;
    const nLayers = 30;
    const paramTypes = [];
    for (let i = 0; i < nLayers; i++) {
      paramTypes.push(tt([hidden, hidden]));
      paramTypes.push(tt([hidden]));
      paramTypes.push(tt([hidden]));
    }

    const func = buildFunction('deep_mlp',
      [tt([1, hidden]), ...paramTypes],
      [tt([1, hidden])],
      (b, args) => {
        let x = args[0];
        let pi = 1;

        for (let i = 0; i < nLayers; i++) {
          const w = args[pi++], g = args[pi++], bt = args[pi++];
          const xr = b.reshape(x, [hidden]).getResult(0);
          const h = b.matmul(b.reshape(x, [1, hidden]).getResult(0), w).getResult(0);

          if (i % 3 === 0) {
            x = b.gelu(h).getResult(0);
          } else if (i % 3 === 1) {
            x = b.silu(h).getResult(0);
          } else {
            x = b.relu(h).getResult(0);
          }

          x = b.layernorm(b.reshape(x, [1, 1, hidden]).getResult(0), g, bt, 2).getResult(0);
          x = b.reshape(x, [1, hidden]).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(200);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('deep_mlp');
    const stats = analyzeKernel(src);
    const info = report('Deep MLP (30 layers)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('Inception-like multi-branch with 8 modules', () => {
    const C = 16, S = 8;

    const func = buildFunction('inception',
      [tt([1, C, S, S])],
      [tt([1, C * 4, S, S])],
      (b, args) => {
        let x = args[0];

        for (let mod = 0; mod < 8; mod++) {
          const inC = mod === 0 ? C : C * 4;

          const w1x1 = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const branch1 = b.relu(b.conv(x, w1x1, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);

          const w3x3r = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const reduce3 = b.relu(b.conv(x, w3x3r, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);
          const w3x3 = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const branch3 = b.relu(b.conv(reduce3, w3x3, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);

          const w5x5r = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const reduce5 = b.relu(b.conv(x, w5x5r, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);
          const w5x5a = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const conv5a = b.relu(b.conv(reduce5, w5x5a, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);
          const w5x5b = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const branch5 = b.relu(b.conv(conv5a, w5x5b, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);

          const wPool = b.constant(0.01, tt([C, inC, 1, 1])).getResult(0);
          const branchPool = b.relu(b.conv(x, wPool, [1, 1], [[0, 0], [0, 0]]).getResult(0)).getResult(0);

          x = b.concat([branch1, branch3, branch5, branchPool], 1).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(250);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('inception');
    const stats = analyzeKernel(src);
    const info = report('Inception (8 modules)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('encoder-decoder with cross-attention (seq2seq)', () => {
    const seqLen = 8, dModel = 32, nLayers = 4;

    const paramTypes = [];
    for (let i = 0; i < nLayers; i++) {
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
    }
    for (let i = 0; i < nLayers; i++) {
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
    }

    const func = buildFunction('seq2seq',
      [tt([seqLen, dModel]), tt([seqLen, dModel]), ...paramTypes],
      [tt([seqLen, dModel])],
      (b, args) => {
        let encOut = args[0];
        let decIn = args[1];
        let pi = 2;

        for (let layer = 0; layer < nLayers; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g = args[pi++], bt = args[pi++];

          const Q = b.matmul(encOut, wQ).getResult(0);
          const K = b.matmul(encOut, wK).getResult(0);
          const V = b.matmul(encOut, wV).getResult(0);
          const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(dModel), F32).getResult(0),
            [seqLen, seqLen], []
          ).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
          const out = b.matmul(attn, V).getResult(0);
          const projected = b.matmul(out, wO).getResult(0);
          const res = b.add(encOut, projected).getResult(0);
          encOut = b.layernorm(
            b.reshape(res, [1, seqLen, dModel]).getResult(0), g, bt, 2
          ).getResult(0);
          encOut = b.reshape(encOut, [seqLen, dModel]).getResult(0);
        }

        let x = decIn;
        for (let layer = 0; layer < nLayers; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], bt1 = args[pi++];
          const cwQ = args[pi++], cwK = args[pi++], cwV = args[pi++], cwO = args[pi++];
          const g2 = args[pi++], bt2 = args[pi++];

          const Q = b.matmul(x, wQ).getResult(0);
          const K = b.matmul(x, wK).getResult(0);
          const V = b.matmul(x, wV).getResult(0);
          const s1 = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale1 = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(dModel), F32).getResult(0),
            [seqLen, seqLen], []
          ).getResult(0);
          const a1 = b.softmax(b.mul(s1, scale1).getResult(0), 1).getResult(0);
          const selfOut = b.matmul(a1, V).getResult(0);
          const p1 = b.matmul(selfOut, wO).getResult(0);
          const r1 = b.add(x, p1).getResult(0);
          const ln1 = b.layernorm(
            b.reshape(r1, [1, seqLen, dModel]).getResult(0), g1, bt1, 2
          ).getResult(0);
          const ln1r = b.reshape(ln1, [seqLen, dModel]).getResult(0);

          const cQ = b.matmul(ln1r, cwQ).getResult(0);
          const cK = b.matmul(encOut, cwK).getResult(0);
          const cV = b.matmul(encOut, cwV).getResult(0);
          const cs = b.matmul(cQ, b.transpose(cK, [1, 0]).getResult(0)).getResult(0);
          const cScale = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(dModel), F32).getResult(0),
            [seqLen, seqLen], []
          ).getResult(0);
          const cAttn = b.softmax(b.mul(cs, cScale).getResult(0), 1).getResult(0);
          const crossOut = b.matmul(cAttn, cV).getResult(0);
          const cp = b.matmul(crossOut, cwO).getResult(0);
          const r2 = b.add(ln1r, cp).getResult(0);
          x = b.layernorm(
            b.reshape(r2, [1, seqLen, dModel]).getResult(0), g2, bt2, 2
          ).getResult(0);
          x = b.reshape(x, [seqLen, dModel]).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(150);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('seq2seq');
    const stats = analyzeKernel(src);
    const info = report('Seq2Seq (4L enc + 4L dec+cross)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(60000);
    expect(info.quality).toHaveLength(0);
  });

  it('wide ResNeXt with grouped conv (16 groups, 10 blocks)', () => {
    const C = 64, S = 8, groups = 16;

    const func = buildFunction('resnext',
      [tt([1, C, S, S])],
      [tt([1, C, S, S])],
      (b, args) => {
        let x = args[0];

        for (let block = 0; block < 10; block++) {
          const w1 = b.constant(0.01, tt([C, C, 1, 1])).getResult(0);
          const r1 = b.relu(b.conv(x, w1, [1, 1], [[0, 0], [0, 0]]).getResult(0));

          const wg = b.constant(0.01, tt([C, C / groups, 3, 3])).getResult(0);
          const gc = b.conv(r1.getResult(0), wg, [1, 1], [[1, 1], [1, 1]], { groups });
          const r2 = b.relu(gc.getResult(0));

          const w3 = b.constant(0.01, tt([C, C, 1, 1])).getResult(0);
          const out = b.conv(r2.getResult(0), w3, [1, 1], [[0, 0], [0, 0]]);

          const g = b.constant(1.0, tt([C])).getResult(0);
          const bt = b.constant(0.0, tt([C])).getResult(0);
          const m = b.constant(0.0, tt([C])).getResult(0);
          const v = b.constant(1.0, tt([C])).getResult(0);
          const bn = b.batchnorm(out.getResult(0), g, bt, m, v);

          const res = b.add(bn.getResult(0), x);
          x = b.relu(res.getResult(0)).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(200);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('resnext');
    const stats = analyzeKernel(src);
    const info = report('ResNeXt (10 blocks, 16 groups)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('multi-scale feature pyramid (FPN) with 5 scales', () => {
    const func = buildFunction('fpn',
      [
        tt([1, 16, 32, 32]),
        tt([1, 32, 16, 16]),
        tt([1, 64, 8, 8]),
        tt([1, 128, 4, 4]),
        tt([1, 256, 2, 2]),
      ],
      [
        tt([1, 64, 32, 32]),
        tt([1, 64, 16, 16]),
        tt([1, 64, 8, 8]),
        tt([1, 64, 4, 4]),
        tt([1, 64, 2, 2]),
      ],
      (b, args) => {
        const channels = [16, 32, 64, 128, 256];
        const spatials = [32, 16, 8, 4, 2];
        const outC = 64;

        const laterals = [];
        for (let i = 0; i < 5; i++) {
          const w = b.constant(0.01, tt([outC, channels[i], 1, 1])).getResult(0);
          laterals.push(b.conv(args[i], w, [1, 1], [[0, 0], [0, 0]]).getResult(0));
        }

        const outputs = new Array(5);
        outputs[4] = laterals[4];

        for (let i = 3; i >= 0; i--) {
          const upsampled = b.resize(outputs[i + 1], [spatials[i], spatials[i]], 'nearest', { layout: 'NCHW' });
          const merged = b.add(laterals[i], upsampled.getResult(0));
          const w = b.constant(0.01, tt([outC, outC, 3, 3])).getResult(0);
          outputs[i] = b.conv(merged.getResult(0), w, [1, 1], [[1, 1], [1, 1]]).getResult(0);
        }

        b.returnOp(outputs);
      }
    );

    const opCount = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('fpn');
    const stats = analyzeKernel(src);
    const info = report('FPN (5 scales)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('BERT-like model (embeddings + 4 transformer layers + pooler)', () => {
    const seqLen = 16, dModel = 64, dFF = 128, vocabSize = 128;

    const paramTypes = [
      tt([vocabSize, dModel]),
    ];
    for (let i = 0; i < 4; i++) {
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel, dFF]));
      paramTypes.push(tt([dFF, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
    }
    paramTypes.push(tt([dModel, dModel]));

    const func = buildFunction('bert',
      [tt([seqLen]), ...paramTypes],
      [tt([dModel])],
      (b, args) => {
        let pi = 1;
        const embedW = args[pi++];

        const embedded = b.embedding(embedW, args[0]);
        let x = embedded.getResult(0);

        for (let layer = 0; layer < 4; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], bt1 = args[pi++];
          const wFF1 = args[pi++], wFF2 = args[pi++];
          const g2 = args[pi++], bt2 = args[pi++];

          const Q = b.matmul(x, wQ).getResult(0);
          const K = b.matmul(x, wK).getResult(0);
          const V = b.matmul(x, wV).getResult(0);

          const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(dModel), F32).getResult(0),
            [seqLen, seqLen], []
          ).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
          const attnOut = b.matmul(attn, V).getResult(0);
          const proj = b.matmul(attnOut, wO).getResult(0);

          const res1 = b.add(x, proj).getResult(0);
          const ln1 = b.layernorm(
            b.reshape(res1, [1, seqLen, dModel]).getResult(0), g1, bt1, 2
          ).getResult(0);
          const ln1r = b.reshape(ln1, [seqLen, dModel]).getResult(0);

          const ff1 = b.gelu(b.matmul(ln1r, wFF1).getResult(0)).getResult(0);
          const ff2 = b.matmul(ff1, wFF2).getResult(0);
          const res2 = b.add(ln1r, ff2).getResult(0);
          x = b.layernorm(
            b.reshape(res2, [1, seqLen, dModel]).getResult(0), g2, bt2, 2
          ).getResult(0);
          x = b.reshape(x, [seqLen, dModel]).getResult(0);
        }

        const poolerW = args[pi++];
        const cls = b.slice(x, [0, 0], [1, dModel]).getResult(0);
        const clsFlat = b.reshape(cls, [1, dModel]).getResult(0);
        const pooled = b.matmul(clsFlat, poolerW).getResult(0);
        const out = b.tanh(pooled).getResult(0);
        const outFlat = b.reshape(out, [dModel]).getResult(0);

        b.returnOp([outFlat]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(80);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('bert');
    const stats = analyzeKernel(src);
    const info = report('BERT (4 layers)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('MobileNetV2-like with inverted residuals (15 blocks)', () => {
    const func = buildFunction('mobilenetv2',
      [tt([1, 16, 16, 16])],
      [tt([1, 16, 16, 16])],
      (b, args) => {
        let x = args[0];
        const C = 16;

        for (let block = 0; block < 15; block++) {
          const expand = C * 4;
          const wExpand = b.constant(0.01, tt([expand, C, 1, 1])).getResult(0);
          const ex = b.silu(b.conv(x, wExpand, [1, 1], [[0, 0], [0, 0]]).getResult(0));

          const wDW = b.constant(0.01, tt([expand, 1, 3, 3])).getResult(0);
          const dw = b.silu(b.conv(ex.getResult(0), wDW, [1, 1], [[1, 1], [1, 1]], { groups: expand }).getResult(0));

          const wProj = b.constant(0.01, tt([C, expand, 1, 1])).getResult(0);
          const proj = b.conv(dw.getResult(0), wProj, [1, 1], [[0, 0], [0, 0]]);

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

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(200);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('mobilenetv2');
    const stats = analyzeKernel(src);
    const info = report('MobileNetV2 (15 inv. residuals)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('GPT-style causal decoder (8 layers, FFN + attention)', () => {
    const seqLen = 16, dModel = 48, dFF = 96;

    const paramTypes = [];
    for (let i = 0; i < 8; i++) {
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel, dFF]));
      paramTypes.push(tt([dFF, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
    }

    const func = buildFunction('gpt',
      [tt([seqLen, dModel]), ...paramTypes],
      [tt([seqLen, dModel])],
      (b, args) => {
        let x = args[0];
        let pi = 1;

        for (let layer = 0; layer < 8; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], bt1 = args[pi++];
          const wFF1 = args[pi++], wFF2 = args[pi++];
          const g2 = args[pi++], bt2 = args[pi++];

          const ln1 = b.layernorm(
            b.reshape(x, [1, seqLen, dModel]).getResult(0), g1, bt1, 2
          ).getResult(0);
          const ln1r = b.reshape(ln1, [seqLen, dModel]).getResult(0);

          const Q = b.matmul(ln1r, wQ).getResult(0);
          const K = b.matmul(ln1r, wK).getResult(0);
          const V = b.matmul(ln1r, wV).getResult(0);
          const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(dModel), F32).getResult(0),
            [seqLen, seqLen], []
          ).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
          const attnOut = b.matmul(attn, V).getResult(0);
          const proj = b.matmul(attnOut, wO).getResult(0);
          x = b.add(x, proj).getResult(0);

          const ln2 = b.layernorm(
            b.reshape(x, [1, seqLen, dModel]).getResult(0), g2, bt2, 2
          ).getResult(0);
          const ln2r = b.reshape(ln2, [seqLen, dModel]).getResult(0);

          const ff1 = b.gelu(b.matmul(ln2r, wFF1).getResult(0)).getResult(0);
          const ff2 = b.matmul(ff1, wFF2).getResult(0);
          x = b.add(x, ff2).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(150);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('gpt');
    const stats = analyzeKernel(src);
    const info = report('GPT (8 layers)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(60000);
    expect(info.quality).toHaveLength(0);
  });

  it('EfficientNet-like compound scaling (width + depth + resolution)', () => {
    const C = 24, S = 16;

    const func = buildFunction('efficientnet',
      [tt([1, 3, S, S])],
      [tt([1, C * 4, S / 2, S / 2])],
      (b, args) => {
        let x = args[0];

        const wStem = b.constant(0.01, tt([C, 3, 3, 3])).getResult(0);
        x = b.silu(b.conv(x, wStem, [1, 1], [[1, 1], [1, 1]]).getResult(0)).getResult(0);

        const stages = [
          { blocks: 2, inC: C, outC: C, stride: 1 },
          { blocks: 3, inC: C, outC: C * 2, stride: 2 },
          { blocks: 3, inC: C * 2, outC: C * 4, stride: 1 },
        ];

        for (const stage of stages) {
          for (let blk = 0; blk < stage.blocks; blk++) {
            const inC = blk === 0 ? stage.inC : stage.outC;
            const stride = blk === 0 ? stage.stride : 1;
            const pad = stride === 2 ? [[0, 1], [0, 1]] : [[1, 1], [1, 1]];

            const expand = inC * 4;
            const wE = b.constant(0.01, tt([expand, inC, 1, 1])).getResult(0);
            const ex = b.silu(b.conv(x, wE, [1, 1], [[0, 0], [0, 0]]).getResult(0));

            const wDW = b.constant(0.01, tt([expand, 1, 3, 3])).getResult(0);
            const dw = b.silu(b.conv(ex.getResult(0), wDW, [stride, stride], pad, { groups: expand }).getResult(0));

            const squeezed = stage.outC;
            const seC = Math.max(1, Math.floor(squeezed / 4));
            const wSE1 = b.constant(0.01, tt([seC, expand, 1, 1])).getResult(0);
            const wSE2 = b.constant(0.01, tt([expand, seC, 1, 1])).getResult(0);

            const dwOut = dw.getResult(0);
            const dwShape = dwOut.type.shape;
            const pooled = b.reduce(
              dwOut,
              b.scalarConstant(0, F32).getResult(0),
              [2, 3],
              'mean'
            );
            const poolR = b.reshape(pooled.getResult(0), [1, expand, 1, 1]).getResult(0);
            const se1 = b.silu(b.conv(poolR, wSE1, [1, 1], [[0, 0], [0, 0]]).getResult(0));
            const se2 = b.sigmoid(b.conv(se1.getResult(0), wSE2, [1, 1], [[0, 0], [0, 0]]).getResult(0));

            const seBcast = b.broadcast(
              b.reshape(se2.getResult(0), [1, expand, 1, 1]).getResult(0),
              dwShape, [0, 1, 2, 3]
            );
            const scaled = b.mul(dwOut, seBcast.getResult(0));

            const wProj = b.constant(0.01, tt([stage.outC, expand, 1, 1])).getResult(0);
            const proj = b.conv(scaled.getResult(0), wProj, [1, 1], [[0, 0], [0, 0]]);

            if (inC === stage.outC && stride === 1) {
              x = b.add(proj.getResult(0), x).getResult(0);
            } else {
              x = proj.getResult(0);
            }
          }
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(150);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('efficientnet');
    const stats = analyzeKernel(src);
    const info = report('EfficientNet (3 stages)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('vision transformer (ViT) with patch embedding', () => {
    const patchSize = 4, imgSize = 16, dModel = 48;
    const nPatches = (imgSize / patchSize) ** 2;

    const paramTypes = [
      tt([dModel, 3, patchSize, patchSize]),
    ];
    for (let i = 0; i < 6; i++) {
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel, dModel * 2]));
      paramTypes.push(tt([dModel * 2, dModel]));
      paramTypes.push(tt([dModel]));
      paramTypes.push(tt([dModel]));
    }

    const func = buildFunction('vit',
      [tt([1, 3, imgSize, imgSize]), ...paramTypes],
      [tt([nPatches, dModel])],
      (b, args) => {
        let pi = 1;
        const patchW = args[pi++];

        const patches = b.conv(args[0], patchW, [patchSize, patchSize], [[0, 0], [0, 0]]);
        let x = b.reshape(patches.getResult(0), [nPatches, dModel]).getResult(0);

        for (let layer = 0; layer < 6; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], bt1 = args[pi++];
          const wFF1 = args[pi++], wFF2 = args[pi++];
          const g2 = args[pi++], bt2 = args[pi++];

          const ln1 = b.layernorm(
            b.reshape(x, [1, nPatches, dModel]).getResult(0), g1, bt1, 2
          ).getResult(0);
          const ln1r = b.reshape(ln1, [nPatches, dModel]).getResult(0);

          const Q = b.matmul(ln1r, wQ).getResult(0);
          const K = b.matmul(ln1r, wK).getResult(0);
          const V = b.matmul(ln1r, wV).getResult(0);
          const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(dModel), F32).getResult(0),
            [nPatches, nPatches], []
          ).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
          const attnOut = b.matmul(attn, V).getResult(0);
          const proj = b.matmul(attnOut, wO).getResult(0);
          x = b.add(x, proj).getResult(0);

          const ln2 = b.layernorm(
            b.reshape(x, [1, nPatches, dModel]).getResult(0), g2, bt2, 2
          ).getResult(0);
          const ln2r = b.reshape(ln2, [nPatches, dModel]).getResult(0);
          const ff1 = b.gelu(b.matmul(ln2r, wFF1).getResult(0)).getResult(0);
          const ff2 = b.matmul(ff1, wFF2).getResult(0);
          x = b.add(x, ff2).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(100);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('vit');
    const stats = analyzeKernel(src);
    const info = report('ViT (6 layers, 4x4 patches)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('Mixture-of-Experts with 4 experts and gating', () => {
    const seqLen = 8, dModel = 32, dFF = 64, nExperts = 4;

    const paramTypes = [
      tt([dModel, nExperts]),
    ];
    for (let e = 0; e < nExperts; e++) {
      paramTypes.push(tt([dModel, dFF]));
      paramTypes.push(tt([dFF, dModel]));
    }

    const func = buildFunction('moe',
      [tt([seqLen, dModel]), ...paramTypes],
      [tt([seqLen, dModel])],
      (b, args) => {
        const x = args[0];
        let pi = 1;
        const gateW = args[pi++];

        const gateLogits = b.matmul(x, gateW).getResult(0);
        const gateProbs = b.softmax(gateLogits, 1).getResult(0);

        const expertOutputs = [];
        for (let e = 0; e < nExperts; e++) {
          const wUp = args[pi++], wDown = args[pi++];
          const h = b.gelu(b.matmul(x, wUp).getResult(0)).getResult(0);
          const out = b.matmul(h, wDown).getResult(0);
          expertOutputs.push(out);
        }

        let combined = b.mul(
          expertOutputs[0],
          b.broadcast(
            b.slice(gateProbs, [0, 0], [seqLen, 1]).getResult(0),
            [seqLen, dModel], [0, 1]
          ).getResult(0)
        ).getResult(0);

        for (let e = 1; e < nExperts; e++) {
          const weight = b.broadcast(
            b.slice(gateProbs, [0, e], [seqLen, e + 1]).getResult(0),
            [seqLen, dModel], [0, 1]
          ).getResult(0);
          const weighted = b.mul(expertOutputs[e], weight).getResult(0);
          combined = b.add(combined, weighted).getResult(0);
        }

        const res = b.add(x, combined).getResult(0);
        b.returnOp([res]);
      }
    );

    const opCount = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('moe');
    const stats = analyzeKernel(src);
    const info = report('MoE (4 experts)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('SwinTransformer-inspired with windowed attention (4 layers)', () => {
    const H = 8, W = 8, C = 32, windowSize = 4;
    const nWindows = (H / windowSize) * (W / windowSize);

    const paramTypes = [];
    for (let i = 0; i < 4; i++) {
      paramTypes.push(tt([C, C]));
      paramTypes.push(tt([C, C]));
      paramTypes.push(tt([C, C]));
      paramTypes.push(tt([C, C]));
      paramTypes.push(tt([C]));
      paramTypes.push(tt([C]));
      paramTypes.push(tt([C, C * 2]));
      paramTypes.push(tt([C * 2, C]));
      paramTypes.push(tt([C]));
      paramTypes.push(tt([C]));
    }

    const func = buildFunction('swin',
      [tt([1, C, H, W]), ...paramTypes],
      [tt([1, C, H, W])],
      (b, args) => {
        let x = args[0];
        let pi = 1;

        for (let layer = 0; layer < 4; layer++) {
          const wQ = args[pi++], wK = args[pi++], wV = args[pi++], wO = args[pi++];
          const g1 = args[pi++], bt1 = args[pi++];
          const wFF1 = args[pi++], wFF2 = args[pi++];
          const g2 = args[pi++], bt2 = args[pi++];

          const xFlat = b.reshape(x, [H * W, C]).getResult(0);
          const ln1 = b.layernorm(
            b.reshape(xFlat, [1, H * W, C]).getResult(0), g1, bt1, 2
          ).getResult(0);
          const ln1r = b.reshape(ln1, [H * W, C]).getResult(0);

          const Q = b.matmul(ln1r, wQ).getResult(0);
          const K = b.matmul(ln1r, wK).getResult(0);
          const V = b.matmul(ln1r, wV).getResult(0);

          const scores = b.matmul(Q, b.transpose(K, [1, 0]).getResult(0)).getResult(0);
          const scale = b.broadcast(
            b.scalarConstant(1.0 / Math.sqrt(C), F32).getResult(0),
            [H * W, H * W], []
          ).getResult(0);
          const attn = b.softmax(b.mul(scores, scale).getResult(0), 1).getResult(0);
          const attnOut = b.matmul(attn, V).getResult(0);
          const proj = b.matmul(attnOut, wO).getResult(0);

          const projR = b.reshape(proj, [1, C, H, W]).getResult(0);
          x = b.add(x, projR).getResult(0);

          const xFlat2 = b.reshape(x, [H * W, C]).getResult(0);
          const ln2 = b.layernorm(
            b.reshape(xFlat2, [1, H * W, C]).getResult(0), g2, bt2, 2
          ).getResult(0);
          const ln2r = b.reshape(ln2, [H * W, C]).getResult(0);

          const ff1 = b.gelu(b.matmul(ln2r, wFF1).getResult(0)).getResult(0);
          const ff2 = b.matmul(ff1, wFF2).getResult(0);
          const ff2r = b.reshape(ff2, [1, C, H, W]).getResult(0);
          x = b.add(x, ff2r).getResult(0);
        }

        b.returnOp([x]);
      }
    );

    const opCount = countOps(func);
    expect(opCount).toBeGreaterThan(80);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('swin');
    const stats = analyzeKernel(src);
    const info = report('Swin Transformer (4 layers)', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('multi-task detection + segmentation head', () => {
    const C = 32, S = 8;
    const nClasses = 10, nAnchors = 3;

    const func = buildFunction('multi_task',
      [tt([1, C, S, S])],
      [
        tt([1, nAnchors * 4, S, S]),
        tt([1, nAnchors * nClasses, S, S]),
        tt([1, 1, S, S]),
      ],
      (b, args) => {
        let x = args[0];

        for (let i = 0; i < 4; i++) {
          const w = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
          const g = b.constant(1.0, tt([C])).getResult(0);
          const bt = b.constant(0.0, tt([C])).getResult(0);
          const m = b.constant(0.0, tt([C])).getResult(0);
          const v = b.constant(1.0, tt([C])).getResult(0);
          const conv = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
          const bn = b.batchnorm(conv.getResult(0), g, bt, m, v);
          x = b.relu(bn.getResult(0)).getResult(0);
        }

        const wBox = b.constant(0.01, tt([nAnchors * 4, C, 1, 1])).getResult(0);
        const boxPred = b.conv(x, wBox, [1, 1], [[0, 0], [0, 0]]);

        const wCls = b.constant(0.01, tt([nAnchors * nClasses, C, 1, 1])).getResult(0);
        const clsPred = b.conv(x, wCls, [1, 1], [[0, 0], [0, 0]]);

        const wSeg = b.constant(0.01, tt([1, C, 1, 1])).getResult(0);
        const segPred = b.sigmoid(b.conv(x, wSeg, [1, 1], [[0, 0], [0, 0]]).getResult(0));

        b.returnOp([boxPred.getResult(0), clsPred.getResult(0), segPred.getResult(0)]);
      }
    );

    const opCount = countOps(func);

    const t0 = performance.now();
    const r = compile(func);
    const compileMs = performance.now() - t0;

    const src = r.getSource('multi_task');
    const stats = analyzeKernel(src);
    const info = report('Multi-task det+seg', opCount, compileMs, stats);

    expect(compileMs).toBeLessThan(30000);
    expect(info.quality).toHaveLength(0);
  });

  it('compile time scales sub-linearly with op count', () => {
    const times = [];
    const opCounts = [];

    for (const nBlocks of [2, 5, 10, 20]) {
      const C = 32, S = 8;
      const func = buildFunction(`scale_${nBlocks}`,
        [tt([1, C, S, S])], [tt([1, C, S, S])],
        (b, args) => {
          let x = args[0];
          for (let i = 0; i < nBlocks; i++) {
            const w = b.constant(0.01, tt([C, C, 3, 3])).getResult(0);
            const g = b.constant(1.0, tt([C])).getResult(0);
            const bt = b.constant(0.0, tt([C])).getResult(0);
            const m = b.constant(0.0, tt([C])).getResult(0);
            const v = b.constant(1.0, tt([C])).getResult(0);
            const conv = b.conv(x, w, [1, 1], [[1, 1], [1, 1]]);
            const bn = b.batchnorm(conv.getResult(0), g, bt, m, v);
            const r = b.relu(bn.getResult(0));
            const res = b.add(r.getResult(0), x);
            x = res.getResult(0);
          }
          b.returnOp([x]);
        }
      );
      const ops = countOps(func);
      opCounts.push(ops);
      const t0 = performance.now();
      compile(func);
      times.push(performance.now() - t0);
    }

    const opsRatio = opCounts[3] / opCounts[0];
    const timeRatio = times[3] / times[0];
    console.log(`  [Scaling] ops ratio: ${opsRatio.toFixed(1)}x, time ratio: ${timeRatio.toFixed(1)}x`);
    console.log(`  [Scaling] ${opCounts.map((o, i) => `${o}ops=${times[i].toFixed(0)}ms`).join(', ')}`);

    expect(timeRatio).toBeLessThan(opsRatio * 2);
  });
});
