import { describe, it, expect } from 'vitest';
import { tensor, compile, compileWithBackward, CPUTarget, CUDATarget } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import * as T from '../../src/tensor/ops/ops.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomTensor, flat } from '../_utils/tensor_data.js';

const rng = mulberry32(1234);
const rt = (shape, lo = -1, hi = 1) => randomTensor(rng, shape, lo, hi);
const ids = (shape, V) => randomTensor(rng, shape, 0, V - 1, 'i32');
const relu = nn.F.relu;

function evalMod(...ms) { ms.forEach((m) => m.eval()); return ms; }

function upsample2x(x) {
  const [B, C, H, W] = x.shape;
  let u = T.reshape(x, [B, C, H, W, 1]);
  u = T.repeat(u, [1, 1, 1, 1, 2]);
  u = T.reshape(u, [B, C, H, W * 2, 1]);
  u = T.repeat(u, [1, 1, 1, 1, 2]);
  u = T.reshape(u, [B, C, H, W * 2, 2]);
  u = T.transpose(u, 3, 4);
  return T.reshape(u, [B, C, H * 2, W * 2]);
}

const MODELS = [
  {
    name: 'MLP (4 x Linear + GELU)',
    build: () => {
      const ls = [new nn.Linear(16, 32), new nn.Linear(32, 32), new nn.Linear(32, 16), new nn.Linear(16, 4)];
      return { fwd: (x) => ls.reduce((h, l, i) => (i < 3 ? nn.F.gelu(l.forward(h)) : l.forward(h)), x), inputs: [rt([4, 16])] };
    },
  },
  {
    name: 'LeNet CNN (conv+pool+fc)',
    build: () => {
      const c1 = new nn.Conv2d(1, 6, 5, { padding: 2 }), c2 = new nn.Conv2d(6, 16, 5);
      const p = new nn.MaxPool2d(2), fl = new nn.Flatten();
      const f1 = new nn.Linear(16 * 5 * 5, 20), f2 = new nn.Linear(20, 10);
      return { fwd: (x) => f2.forward(relu(f1.forward(fl.forward(p.forward(relu(c2.forward(p.forward(relu(c1.forward(x)))))))))), inputs: [rt([2, 1, 28, 28])] };
    },
  },
  {
    name: 'VGG-style deep CNN (6 conv + BN)',
    build: () => {
      const chans = [3, 8, 8, 16, 16, 32, 32];
      const convs = [], bns = [];
      for (let i = 0; i < 6; i++) { convs.push(new nn.Conv2d(chans[i], chans[i + 1], 3, { padding: 1 })); bns.push(new nn.BatchNorm2d(chans[i + 1])); }
      evalMod(...bns);
      const p = new nn.MaxPool2d(2), fl = new nn.Flatten(), fc = new nn.Linear(32 * 4 * 4, 10);
      return {
        fwd: (x) => { let h = x; for (let i = 0; i < 6; i++) { h = relu(bns[i].forward(convs[i].forward(h))); if (i % 2 === 1) h = p.forward(h); } return fc.forward(fl.forward(h)); },
        inputs: [rt([2, 3, 32, 32])],
      };
    },
  },
  {
    name: 'ResNet blocks (residual add + 1x1 downsample)',
    build: () => {
      const mk = (cin, cout, stride) => ({
        c1: new nn.Conv2d(cin, cout, 3, { stride, padding: 1 }), b1: new nn.BatchNorm2d(cout),
        c2: new nn.Conv2d(cout, cout, 3, { padding: 1 }), b2: new nn.BatchNorm2d(cout),
        ds: cin !== cout || stride !== 1 ? { c: new nn.Conv2d(cin, cout, 1, { stride }), b: new nn.BatchNorm2d(cout) } : null,
      });
      const blocks = [mk(8, 8, 1), mk(8, 16, 2), mk(16, 16, 1)];
      blocks.forEach((b) => { evalMod(b.b1, b.b2); if (b.ds) evalMod(b.ds.b); });
      const stem = new nn.Conv2d(3, 8, 3, { padding: 1 }), sbn = new nn.BatchNorm2d(8);
      evalMod(sbn);
      const gap = new nn.AdaptiveAvgPool2d(1), fl = new nn.Flatten(), fc = new nn.Linear(16, 10);
      return {
        fwd: (x) => {
          let h = relu(sbn.forward(stem.forward(x)));
          for (const b of blocks) {
            const idn = b.ds ? b.ds.b.forward(b.ds.c.forward(h)) : h;
            let y = relu(b.b1.forward(b.c1.forward(h)));
            y = b.b2.forward(b.c2.forward(y));
            h = relu(T.add(y, idn));
          }
          return fc.forward(fl.forward(gap.forward(h)));
        },
        inputs: [rt([2, 3, 16, 16])],
      };
    },
  },
  {
    name: 'MobileNet depthwise-separable (grouped conv)',
    build: () => {
      const dw1 = new nn.Conv2d(8, 8, 3, { padding: 1, groups: 8 }), pw1 = new nn.Conv2d(8, 16, 1);
      const dw2 = new nn.Conv2d(16, 16, 3, { padding: 1, stride: 2, groups: 16 }), pw2 = new nn.Conv2d(16, 32, 1);
      const b1 = new nn.BatchNorm2d(16), b2 = new nn.BatchNorm2d(32);
      evalMod(b1, b2);
      const stem = new nn.Conv2d(3, 8, 3, { padding: 1 });
      const gap = new nn.AdaptiveAvgPool2d(1), fl = new nn.Flatten(), fc = new nn.Linear(32, 10);
      return {
        fwd: (x) => {
          let h = relu(stem.forward(x));
          h = relu(b1.forward(pw1.forward(dw1.forward(h))));
          h = relu(b2.forward(pw2.forward(dw2.forward(h))));
          return fc.forward(fl.forward(gap.forward(h)));
        },
        inputs: [rt([2, 3, 16, 16])],
      };
    },
  },
  {
    name: 'GPT-2 decoder (causal MHA + pre-norm)',
    build: () => {
      const V = 32, D = 16, SEQ = 6;
      const tok = new nn.Embedding(V, D), pos = new nn.Embedding(SEQ, D);
      const blocks = Array.from({ length: 2 }, () => new nn.TransformerEncoderLayer(D, 2, 32, 0.0, 'gelu', 1e-5, true, true));
      evalMod(...blocks);
      const lnf = new nn.LayerNorm(D, 1e-5), head = new nn.Linear(D, V);
      const p = tensor(Array.from({ length: SEQ }, (_, i) => i));
      return {
        fwd: (x) => { let h = T.add(tok.forward(x), pos.forward(p)); for (const b of blocks) h = b.forward(h, null, null, true); return head.forward(lnf.forward(h)); },
        inputs: [ids([2, SEQ], V)],
      };
    },
  },
  {
    name: 'BERT encoder + pooler + classifier',
    build: () => {
      const V = 32, D = 16, SEQ = 6;
      const tok = new nn.Embedding(V, D), pos = new nn.Embedding(SEQ, D), seg = new nn.Embedding(2, D);
      const emln = new nn.LayerNorm(D, 1e-12);
      const enc = new nn.TransformerEncoder(new nn.TransformerEncoderLayer(D, 2, 32, 0.0, 'gelu', 1e-12, true, false), 2);
      evalMod(enc);
      const pool = new nn.Linear(D, D), cls = new nn.Linear(D, 3);
      const p = tensor(Array.from({ length: SEQ }, (_, i) => i));
      const s = tensor(Array.from({ length: SEQ }, () => 0));
      return {
        fwd: (x) => {
          const h = enc.forward(emln.forward(T.add(T.add(tok.forward(x), pos.forward(p)), seg.forward(s))));
          const first = T.reshape(T.slice(h, 1, 0, 1), [h.shape[0], D]);
          return cls.forward(T.tanh(pool.forward(first)));
        },
        inputs: [ids([2, SEQ], V)],
      };
    },
  },
  {
    name: 'ViT (patch embed + transformer + head)',
    build: () => {
      const D = 16, P = 4, IMG = 16, NP = (IMG / P) * (IMG / P);
      const patch = new nn.Conv2d(3, D, P, { stride: P });
      const pos = new nn.Embedding(NP, D);
      const blocks = Array.from({ length: 2 }, () => new nn.TransformerEncoderLayer(D, 2, 32, 0.0, 'gelu', 1e-5, true, true));
      evalMod(...blocks);
      const lnf = new nn.LayerNorm(D, 1e-5), head = new nn.Linear(D, 10);
      const p = tensor(Array.from({ length: NP }, (_, i) => i));
      return {
        fwd: (x) => {
          const e = patch.forward(x);
          const B = e.shape[0];
          let h = T.transpose(T.reshape(e, [B, D, NP]), 1, 2);
          h = T.add(h, pos.forward(p));
          for (const b of blocks) h = b.forward(h);
          return head.forward(T.mean(lnf.forward(h), 1));
        },
        inputs: [rt([2, 3, IMG, IMG])],
      };
    },
  },
  {
    name: 'Transformer encoder-decoder (seq2seq)',
    build: () => {
      const V = 24, D = 16, S = 5;
      const emb = new nn.Embedding(V, D);
      const tr = new nn.Transformer({ dModel: D, nhead: 2, numEncoderLayers: 1, numDecoderLayers: 1, dimFeedforward: 32, dropout: 0.0 });
      evalMod(tr);
      const head = new nn.Linear(D, V);
      return {
        fwd: (src, tgt) => head.forward(tr.forward(emb.forward(src), emb.forward(tgt))),
        inputs: [ids([2, S], V), ids([2, S], V)],
      };
    },
  },
  {
    name: 'BiLSTM classifier (fwd + reversed, concat)',
    build: () => {
      const f = new nn.LSTM(8, 12, 1, true), b = new nn.LSTM(8, 12, 1, true);
      const fc = new nn.Linear(24, 5);
      return {
        fwd: (x) => {
          const hf = f.forward(x)[0];
          const hb = T.flip(b.forward(T.flip(x, [1]))[0], [1]);
          return fc.forward(T.mean(T.cat([hf, hb], 2), 1));
        },
        inputs: [rt([2, 6, 8])],
      };
    },
  },
  {
    name: 'LSTM seq2seq + dot attention',
    build: () => {
      const enc = new nn.LSTM(8, 12, 1, true), dec = new nn.LSTM(8, 12, 1, true);
      const out = new nn.Linear(24, 16);
      return {
        fwd: (src, tgt) => {
          const eo = enc.forward(src)[0];
          const dout = dec.forward(tgt)[0];
          const scores = T.softmax(T.matmul(dout, T.transpose(eo, 1, 2)), 2);
          const ctx = T.matmul(scores, eo);
          return out.forward(T.cat([dout, ctx], 2));
        },
        inputs: [rt([2, 5, 8]), rt([2, 4, 8])],
      };
    },
  },
  {
    name: 'GRU language model',
    build: () => {
      const V = 24, D = 12;
      const emb = new nn.Embedding(V, D), g = new nn.GRU(D, 16, 2, true), head = new nn.Linear(16, V);
      return { fwd: (x) => head.forward(g.forward(emb.forward(x))[0]), inputs: [ids([2, 6], V)] };
    },
  },
  {
    name: 'TextCNN (Conv1d multi-kernel + max-over-time)',
    build: () => {
      const V = 24, D = 12, SEQ = 10;
      const emb = new nn.Embedding(V, D);
      const convs = [3, 4, 5].map((k) => new nn.Conv1d(D, 8, k));
      const fc = new nn.Linear(24, 4);
      return {
        fwd: (x) => {
          const e = T.transpose(emb.forward(x), 1, 2);
          const feats = convs.map((c) => T.max(relu(c.forward(e)), 2)[0] ?? T.max(relu(c.forward(e)), 2));
          return fc.forward(T.cat(feats, 1));
        },
        inputs: [ids([2, SEQ], V)],
      };
    },
  },
  {
    name: 'Conv autoencoder (encode + nearest upsample decode)',
    build: () => {
      const e1 = new nn.Conv2d(1, 8, 3, { padding: 1, stride: 2 }), e2 = new nn.Conv2d(8, 16, 3, { padding: 1, stride: 2 });
      const d1 = new nn.Conv2d(16, 8, 3, { padding: 1 }), d2 = new nn.Conv2d(8, 1, 3, { padding: 1 });
      return {
        fwd: (x) => {
          const h = relu(e2.forward(relu(e1.forward(x))));
          return d2.forward(upsample2x(relu(d1.forward(upsample2x(h)))));
        },
        inputs: [rt([2, 1, 16, 16])],
      };
    },
  },
  {
    name: 'U-Net (skip connections + upsample)',
    build: () => {
      const c1 = new nn.Conv2d(1, 8, 3, { padding: 1 }), c2 = new nn.Conv2d(8, 16, 3, { padding: 1 });
      const bott = new nn.Conv2d(16, 32, 3, { padding: 1 });
      const u1 = new nn.Conv2d(32 + 16, 16, 3, { padding: 1 }), u2 = new nn.Conv2d(16 + 8, 8, 3, { padding: 1 });
      const head = new nn.Conv2d(8, 2, 1), p = new nn.MaxPool2d(2);
      return {
        fwd: (x) => {
          const s1 = relu(c1.forward(x));
          const s2 = relu(c2.forward(p.forward(s1)));
          const b = relu(bott.forward(p.forward(s2)));
          const d1 = relu(u1.forward(T.cat([upsample2x(b), s2], 1)));
          const d2 = relu(u2.forward(T.cat([upsample2x(d1), s1], 1)));
          return head.forward(d2);
        },
        inputs: [rt([2, 1, 16, 16])],
      };
    },
  },
  {
    name: 'DLRM-style (embeddings + concat + MLP)',
    build: () => {
      const embs = [new nn.Embedding(20, 8), new nn.Embedding(15, 8), new nn.Embedding(10, 8)];
      const bot = [new nn.Linear(6, 16), new nn.Linear(16, 8)];
      const top = [new nn.Linear(32, 16), new nn.Linear(16, 1)];
      return {
        fwd: (dense, i0, i1, i2) => {
          const d = bot.reduce((h, l) => relu(l.forward(h)), dense);
          const cats = [embs[0].forward(i0), embs[1].forward(i1), embs[2].forward(i2)];
          return top[1].forward(relu(top[0].forward(T.cat([d, ...cats], 1))));
        },
        inputs: [rt([4, 6]), ids([4], 20), ids([4], 15), ids([4], 10)],
      };
    },
  },
  {
    name: 'Two-tower retrieval (shared encoder + dot score)',
    build: () => {
      const enc = [new nn.Linear(12, 16), new nn.Linear(16, 8)];
      const run = (x) => enc.reduce((h, l, i) => (i === 0 ? relu(l.forward(h)) : l.forward(h)), x);
      return { fwd: (a, b) => T.matmul(run(a), T.transpose(run(b), 0, 1)), inputs: [rt([4, 12]), rt([4, 12])] };
    },
  },
];

function relErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return m;
}

const report = { fwd: [], cuda: [], bwd: [], cudarun: [] };

describe('popular model architectures: compile coverage probe', () => {
  for (const m of MODELS) {
    it(`FWD cpu — ${m.name}`, async () => {
      const { fwd, inputs } = m.build();
      const eager = flat(fwd(...inputs));
      const t0 = performance.now();
      const compiled = compile({ forward: fwd }, inputs, { target: CPUTarget() });
      const out = flat(await compiled(...inputs));
      const ms = performance.now() - t0;
      const err = relErr(eager, out);
      report.fwd.push({ name: m.name, ok: err < 3e-3, err, ms, kernels: compiled.kernels?.().length ?? -1 });
      expect(out.length).toBe(eager.length);
      expect(err).toBeLessThan(3e-3);
    });
  }

  for (const m of MODELS) {
    it(`FWD cuda-codegen — ${m.name}`, () => {
      const { fwd, inputs } = m.build();
      const compiled = compile({ forward: fwd }, inputs, { target: CUDATarget() });
      const src = compiled.source();
      report.cuda.push({ name: m.name, ok: /__global__\s+void/.test(src), kernels: compiled.kernels().length, bytes: src.length });
      expect(src).toMatch(/__global__\s+void/);
      expect(src).not.toMatch(/\bMath\.\w+|\bnew Float32Array\b|=>/);
    });
  }

  for (const m of MODELS) {
    it(`BWD cpu — ${m.name}`, () => {
      const { fwd, inputs } = m.build();
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode: 'separate' });
      const out = cf(...inputs);
      const grads = cf.backward(ones(out.shape));
      const finite = grads.every((g) => g == null || flat(g).every(Number.isFinite));
      report.bwd.push({ name: m.name, ok: grads.length >= inputs.length && finite, n: grads.length });
      expect(grads.length).toBeGreaterThanOrEqual(inputs.length);
      expect(finite, 'a gradient contained NaN/Inf').toBe(true);
    });
  }

  for (const m of MODELS) {
    it(`RUN cuda-device — ${m.name}`, async () => {
      const { fwd, inputs } = m.build();
      const eager = flat(fwd(...inputs));
      const compiled = compile({ forward: fwd }, inputs, { target: CUDATarget() });
      const out = flat(await compiled(...inputs));
      const err = relErr(eager, out);
      report.cudarun.push({ name: m.name, ok: err < 5e-3, err });
      expect(err).toBeLessThan(5e-3);
    });
  }

  it('ZZ summary', () => {
    const line = (r) => `${r.ok ? 'PASS' : 'FAIL'}  ${r.name}`;
    console.log('\n===== FORWARD / CPU (compiled == eager) =====');
    for (const r of report.fwd) console.log(`  ${line(r)}  err=${r.err.toExponential(1)} kernels=${r.kernels} ${r.ms.toFixed(0)}ms`);
    console.log('\n===== FORWARD / CUDA codegen =====');
    for (const r of report.cuda) console.log(`  ${line(r)}  kernels=${r.kernels} src=${(r.bytes / 1024).toFixed(1)}KB`);
    console.log('\n===== BACKWARD / CPU =====');
    for (const r of report.bwd) console.log(`  ${line(r)}  grads=${r.n}`);
    console.log(`\nTOTAL: fwd ${report.fwd.filter((r) => r.ok).length}/${MODELS.length}, cuda ${report.cuda.filter((r) => r.ok).length}/${MODELS.length}, bwd ${report.bwd.filter((r) => r.ok).length}/${MODELS.length}`);
  });
});
