import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { TensorType, ScalarType } from '../../../src/compiler/ir/graph/types.js';
import { buildFunction } from '../../../src/compiler/ir/graph/builder.js';
import { CPUTarget, GPUTarget } from '../../../src/compiler/backend/target.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';
import { RuntimeTensor } from '../../../src/compiler/runtime/runtime.js';

const f32 = ScalarType.F32;
function T(shape) { return new TensorType(shape, f32); }
function rand(n, scale = 0.1) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * scale;
  return a;
}
function randPositive(n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.random() * 0.1 + 0.5;
  return a;
}
function ones(n) { return new Float32Array(n).fill(1); }
function zeros(n) { return new Float32Array(n).fill(0); }
function assertFinite(data, label) {
  for (let i = 0; i < data.length; i++)
    assert.ok(isFinite(data[i]), `${label}[${i}] = ${data[i]}`);
}
function numel(shape) { return shape.reduce((a, b) => a * b, 1); }

describe('Multi-Head Attention', () => {
  it('compiles and runs MHA with 4 heads', () => {
    const B = 2, S = 6, D = 16, H = 4, Dh = D / H;
    const func = buildFunction('mha',
      [T([B, S, D]), T([D, D]), T([D, D]), T([D, D]), T([D, D])],
      [T([B, S, D])],
      (b, [x, wq, wk, wv, wo]) => {
        const flat = b.reshape(x, [B * S, D]);
        const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B, S, D]);
        const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B, S, D]);
        const V = b.reshape(b.matmul(flat.getResult(0), wv).getResult(0), [B, S, D]);
        const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
        const scale = b.broadcast(b.scalarConstant(1.0 / Math.sqrt(Dh), f32).getResult(0), [B, S, S], []);
        const attn = b.softmax(b.mul(scores.getResult(0), scale.getResult(0)).getResult(0), -1);
        const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
        const out = b.reshape(b.matmul(b.reshape(ctx.getResult(0), [B * S, D]).getResult(0), wo).getResult(0), [B, S, D]);
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const inputs = [rand(B*S*D), rand(D*D), rand(D*D), rand(D*D), rand(D*D)];
    const out = RuntimeTensor.zeros([B, S, D]);
    compiled.run('mha', ...inputs.map((d, i) => RuntimeTensor.fromArray(d, func.inputTypes[i].shape)), out);
    assertFinite(out.data, 'mha');
  });
});

describe('Transformer Encoder Block', () => {
  it('compiles attention + FFN + layernorm + residual', () => {
    const B = 1, S = 4, D = 8, Dff = 32;
    const func = buildFunction('encoder_block',
      [T([B,S,D]), T([D,D]), T([D,D]), T([D,D]), T([D,D]),
       T([D]), T([D]), T([D,Dff]), T([Dff,D]), T([D]), T([D])],
      [T([B,S,D])],
      (b, [x, wq, wk, wv, wo, ln1g, ln1b, w1, w2, ln2g, ln2b]) => {
        const flat = b.reshape(x, [B*S, D]);
        const Q = b.reshape(b.matmul(flat.getResult(0), wq).getResult(0), [B,S,D]);
        const K = b.reshape(b.matmul(flat.getResult(0), wk).getResult(0), [B,S,D]);
        const V = b.reshape(b.matmul(flat.getResult(0), wv).getResult(0), [B,S,D]);
        const scores = b.dot(Q.getResult(0), K.getResult(0), [2], [2], [0], [0]);
        const scale = b.broadcast(b.scalarConstant(0.35, f32).getResult(0), [B,S,S], []);
        const attn = b.softmax(b.mul(scores.getResult(0), scale.getResult(0)).getResult(0), -1);
        const ctx = b.dot(attn.getResult(0), V.getResult(0), [2], [1], [0], [0]);
        const proj = b.reshape(b.matmul(b.reshape(ctx.getResult(0), [B*S, D]).getResult(0), wo).getResult(0), [B,S,D]);
        const res1 = b.add(x, proj.getResult(0));
        const ln1 = b.layernorm(res1.getResult(0), ln1g, ln1b, -1, 1e-5);
        const ff1 = b.reshape(b.matmul(b.reshape(ln1.getResult(0), [B*S, D]).getResult(0), w1).getResult(0), [B,S,Dff]);
        const act = b.gelu(ff1.getResult(0));
        const ff2 = b.reshape(b.matmul(b.reshape(act.getResult(0), [B*S, Dff]).getResult(0), w2).getResult(0), [B,S,D]);
        const res2 = b.add(ln1.getResult(0), ff2.getResult(0));
        const ln2 = b.layernorm(res2.getResult(0), ln2g, ln2b, -1, 1e-5);
        b.returnOp([ln2.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B,S,D]);
    compiled.run('encoder_block', ...inputs, out);
    assertFinite(out.data, 'encoder_block');
  });
});

describe('ResNet Bottleneck Block', () => {
  it('compiles conv-bn-relu-conv-bn-relu-conv-bn + skip', () => {
    const B = 1, C = 16, H = 8, W = 8, C4 = 4;
    const func = buildFunction('bottleneck',
      [T([B,C,H,W]),
       T([C4,C,1,1]), T([C4]), T([C4]), T([C4]), T([C4]),
       T([C4,C4,3,3]), T([C4]), T([C4]), T([C4]), T([C4]),
       T([C,C4,1,1]), T([C]), T([C]), T([C]), T([C])],
      [T([B,C,H,W])],
      (b, [x, w1,g1,b1,m1,v1, w2,g2,b2,m2,v2, w3,g3,b3,m3,v3]) => {
        const c1 = b.conv(x, w1, [1,1], [[0,0],[0,0]]);
        const bn1 = b.batchnorm(c1.getResult(0), g1, b1, m1, v1, 1, 1e-5);
        const r1 = b.relu(bn1.getResult(0));
        const c2 = b.conv(r1.getResult(0), w2, [1,1], [[1,1],[1,1]]);
        const bn2 = b.batchnorm(c2.getResult(0), g2, b2, m2, v2, 1, 1e-5);
        const r2 = b.relu(bn2.getResult(0));
        const c3 = b.conv(r2.getResult(0), w3, [1,1], [[0,0],[0,0]]);
        const bn3 = b.batchnorm(c3.getResult(0), g3, b3, m3, v3, 1, 1e-5);
        const skip = b.add(x, bn3.getResult(0));
        const out = b.relu(skip.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const gammaIdx = new Set([2, 7, 12]);
    const betaIdx = new Set([3, 8, 13]);
    const meanIdx = new Set([4, 9, 14]);
    const varIdx = new Set([5, 10, 15]);
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map((s, i) => {
      const n = numel(s);
      if (gammaIdx.has(i)) return RuntimeTensor.fromArray(ones(n), s);
      if (betaIdx.has(i)) return RuntimeTensor.fromArray(zeros(n), s);
      if (meanIdx.has(i)) return RuntimeTensor.fromArray(zeros(n), s);
      if (varIdx.has(i)) return RuntimeTensor.fromArray(ones(n), s);
      return RuntimeTensor.fromArray(rand(n, 0.01), s);
    });
    const out = RuntimeTensor.zeros([B,C,H,W]);
    compiled.run('bottleneck', ...inputs, out);
    assertFinite(out.data, 'bottleneck');
  });
});

describe('MLP Mixer Block', () => {
  it('compiles token-mixing + channel-mixing', () => {
    const B = 2, S = 8, D = 16;
    const func = buildFunction('mixer_block',
      [T([B,S,D]), T([D]), T([D]), T([S,S]), T([D,D]), T([D]), T([D])],
      [T([B,S,D])],
      (b, [x, ln1g, ln1b, wt, wc, ln2g, ln2b]) => {
        const ln1 = b.layernorm(x, ln1g, ln1b, -1, 1e-5);
        const transposed = b.transpose(ln1.getResult(0), [0, 2, 1]);
        const flat = b.reshape(transposed.getResult(0), [B*D, S]);
        const mixed = b.matmul(flat.getResult(0), wt);
        const back = b.transpose(b.reshape(mixed.getResult(0), [B, D, S]).getResult(0), [0, 2, 1]);
        const gelu1 = b.gelu(back.getResult(0));
        const res1 = b.add(x, gelu1.getResult(0));
        const ln2 = b.layernorm(res1.getResult(0), ln2g, ln2b, -1, 1e-5);
        const flat2 = b.reshape(ln2.getResult(0), [B*S, D]);
        const ch = b.matmul(flat2.getResult(0), wc);
        const gelu2 = b.gelu(b.reshape(ch.getResult(0), [B, S, D]).getResult(0));
        const res2 = b.add(res1.getResult(0), gelu2.getResult(0));
        b.returnOp([res2.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B,S,D]);
    compiled.run('mixer_block', ...inputs, out);
    assertFinite(out.data, 'mixer_block');
  });
});

describe('U-Net Encoder-Decoder', () => {
  it('compiles conv downsample + conv upsample + skip connections', () => {
    const B = 1, C = 4, H = 8, W = 8;
    const func = buildFunction('unet_mini',
      [T([B,C,H,W]),
       T([8,C,3,3]), T([16,8,3,3]),
       T([8,16,3,3]), T([C,8,3,3])],
      [T([B,C,H,W])],
      (b, [x, wd1, wd2, wu1, wu2]) => {
        const d1 = b.conv(x, wd1, [1,1], [[1,1],[1,1]]);
        const a1 = b.relu(d1.getResult(0));
        const d2 = b.conv(a1.getResult(0), wd2, [1,1], [[1,1],[1,1]]);
        const a2 = b.relu(d2.getResult(0));
        const u1 = b.conv(a2.getResult(0), wu1, [1,1], [[1,1],[1,1]]);
        const a3 = b.relu(u1.getResult(0));
        const skip = b.add(a3.getResult(0), a1.getResult(0));
        const u2 = b.conv(skip.getResult(0), wu2, [1,1], [[1,1],[1,1]]);
        const out = b.sigmoid(u2.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B,C,H,W]);
    compiled.run('unet_mini', ...inputs, out);
    assertFinite(out.data, 'unet_mini');
    for (let i = 0; i < out.data.length; i++) {
      assert.ok(out.data[i] >= 0 && out.data[i] <= 1, `sigmoid output out of [0,1]: ${out.data[i]}`);
    }
  });
});

describe('LSTM Cell', () => {
  it('compiles forget-input-output gates + cell state update', () => {
    const B = 2, D = 8;
    const func = buildFunction('lstm_cell',
      [T([B,D]), T([B,D]), T([B,D]),
       T([D,4*D]), T([D,4*D]), T([4*D])],
      [T([B,D]), T([B,D])],
      (b, [x, hPrev, cPrev, wx, wh, bias]) => {
        const xProj = b.matmul(x, wx);
        const hProj = b.matmul(hPrev, wh);
        const sum = b.add(xProj.getResult(0), hProj.getResult(0));
        const bcast = b.broadcast(bias, [B, 4*D], [1]);
        const gates = b.add(sum.getResult(0), bcast.getResult(0));
        const i_gate = b.sigmoid(b.slice(gates.getResult(0), [0,0], [B,D]).getResult(0));
        const f_gate = b.sigmoid(b.slice(gates.getResult(0), [0,D], [B,2*D]).getResult(0));
        const g_val = b.tanh(b.slice(gates.getResult(0), [0,2*D], [B,3*D]).getResult(0));
        const o_gate = b.sigmoid(b.slice(gates.getResult(0), [0,3*D], [B,4*D]).getResult(0));
        const fC = b.mul(f_gate.getResult(0), cPrev);
        const iG = b.mul(i_gate.getResult(0), g_val.getResult(0));
        const cNew = b.add(fC.getResult(0), iG.getResult(0));
        const hNew = b.mul(o_gate.getResult(0), b.tanh(cNew.getResult(0)).getResult(0));
        b.returnOp([hNew.getResult(0), cNew.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const outH = RuntimeTensor.zeros([B,D]);
    const outC = RuntimeTensor.zeros([B,D]);
    compiled.run('lstm_cell', ...inputs, outH, outC);
    assertFinite(outH.data, 'lstm_h');
    assertFinite(outC.data, 'lstm_c');
  });
});

describe('GAN Generator Block', () => {
  it('compiles deconv-batchnorm-relu chain', () => {
    const B = 1, C = 8, H = 4, W = 4;
    const func = buildFunction('gen_block',
      [T([B,C,H,W]),
       T([C,C,3,3]), T([C]), T([C]), T([C]), T([C]),
       T([C,C,3,3]), T([C]), T([C]), T([C]), T([C])],
      [T([B,C,H,W])],
      (b, [x, w1,g1,b1,m1,v1, w2,g2,b2,m2,v2]) => {
        const c1 = b.conv(x, w1, [1,1], [[1,1],[1,1]]);
        const bn1 = b.batchnorm(c1.getResult(0), g1, b1, m1, v1, 1, 1e-5);
        const r1 = b.relu(bn1.getResult(0));
        const c2 = b.conv(r1.getResult(0), w2, [1,1], [[1,1],[1,1]]);
        const bn2 = b.batchnorm(c2.getResult(0), g2, b2, m2, v2, 1, 1e-5);
        const out = b.silu(bn2.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { enabled: false } });
    const gIdx = new Set([2, 7]);
    const bIdx = new Set([3, 8]);
    const mIdx = new Set([4, 9]);
    const vIdx = new Set([5, 10]);
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map((s, i) => {
      const n = numel(s);
      if (gIdx.has(i)) return RuntimeTensor.fromArray(ones(n), s);
      if (bIdx.has(i)) return RuntimeTensor.fromArray(zeros(n), s);
      if (mIdx.has(i)) return RuntimeTensor.fromArray(zeros(n), s);
      if (vIdx.has(i)) return RuntimeTensor.fromArray(ones(n), s);
      return RuntimeTensor.fromArray(rand(n, 0.01), s);
    });
    const out = RuntimeTensor.zeros([B,C,H,W]);
    compiled.run('gen_block', ...inputs, out);
    assertFinite(out.data, 'gen_block');
  });
});

describe('Pipeline Stress: All Features Enabled', () => {
  it('compiles transformer with fusion + layout + decomposition', () => {
    const B = 1, S = 4, D = 8;
    const func = buildFunction('full_pipe',
      [T([B,S,D]), T([D,D]), T([D]), T([D])],
      [T([B,S,D])],
      (b, [x, w, lng, lnb]) => {
        const flat = b.reshape(x, [B*S, D]);
        const mm = b.matmul(flat.getResult(0), w);
        const back = b.reshape(mm.getResult(0), [B, S, D]);
        const act = b.gelu(back.getResult(0));
        const res = b.add(x, act.getResult(0));
        const ln = b.layernorm(res.getResult(0), lng, lnb, -1, 1e-5);
        const sm = b.softmax(ln.getResult(0), -1);
        b.returnOp([sm.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), {
      fusion: { enabled: true },
      optimization: { layout: true },
    });
    const shapes = func.inputTypes.map(t => t.shape);
    const inputs = shapes.map(s => RuntimeTensor.fromArray(rand(numel(s)), s));
    const out = RuntimeTensor.zeros([B,S,D]);
    compiled.run('full_pipe', ...inputs, out);
    assertFinite(out.data, 'full_pipe');
    let sum = 0;
    for (let i = 0; i < out.data.length; i++) sum += out.data[i];
    assert.ok(Math.abs(sum - B * S) < 0.1, `softmax rows should sum to ~${B*S}, got ${sum}`);
  });

  it('compiles with dominator fusion strategy', () => {
    const func = buildFunction('dom_strat',
      [T([4,8]), T([8,4])], [T([4,4])],
      (b, [x, w]) => {
        const mm = b.matmul(x, w);
        const e = b.exp(mm.getResult(0));
        const n = b.neg(e.getResult(0));
        const out = b.add(e.getResult(0), n.getResult(0));
        b.returnOp([out.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, CPUTarget(), { fusion: { strategy: 'dominator' } });
    const out = RuntimeTensor.zeros([4,4]);
    compiled.run('dom_strat',
      RuntimeTensor.fromArray(rand(32), [4,8]),
      RuntimeTensor.fromArray(rand(32), [8,4]), out);
    assertFinite(out.data, 'dom_strat');
  });

  it('compiles for GPU target', () => {
    const func = buildFunction('gpu_model',
      [T([2,4,8]), T([8,8]), T([8]), T([8])],
      [T([2,4,8])],
      (b, [x, w, g, beta]) => {
        const flat = b.reshape(x, [8, 8]);
        const mm = b.matmul(flat.getResult(0), w);
        const back = b.reshape(mm.getResult(0), [2, 4, 8]);
        const ln = b.layernorm(back.getResult(0), g, beta, -1, 1e-5);
        const act = b.silu(ln.getResult(0));
        b.returnOp([act.getResult(0)]);
      }
    );
    const compiled = compileGraph(func, GPUTarget());
    const src = compiled.getSource('gpu_model');
    assert.ok(src.includes('__global__'));
    assert.ok(src.length > 500);
  });
});
