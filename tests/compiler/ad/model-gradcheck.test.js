import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import * as T from '../../../src/tensor/ops/ops.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../../_utils/tensor_data.js';

const EPS = 2e-3;
const TOL = 1e-2;
const PROBES_PER_TENSOR = 4;
const MAX_PARAM_TENSORS = 5;

const relu = nn.F.relu;

function evalMod(...ms) { ms.forEach((m) => m.eval()); return ms; }

function upsample2x(x) {
  const [B, C, H, W] = x.shape;
  let u = T.repeat(T.reshape(x, [B, C, H, W, 1]), [1, 1, 1, 1, 2]);
  u = T.repeat(T.reshape(u, [B, C, H, W * 2, 1]), [1, 1, 1, 1, 2]);
  u = T.transpose(T.reshape(u, [B, C, H, W * 2, 2]), 3, 4);
  return T.reshape(u, [B, C, H * 2, W * 2]);
}

const CASES = [
  {
    name: 'CNN (conv + batchnorm + relu + pool + linear)',
    build: (rng) => {
      const c1 = new nn.Conv2d(2, 4, 3, { padding: 1 }), b1 = new nn.BatchNorm2d(4);
      const p = new nn.MaxPool2d(2), fl = new nn.Flatten(), fc = new nn.Linear(4 * 2 * 2, 3);
      evalMod(b1);
      return { fwd: (x) => fc.forward(fl.forward(p.forward(relu(b1.forward(c1.forward(x)))))), shapes: [[1, 2, 4, 4]] };
    },
  },
  {
    name: 'ResNet block (residual add + 1x1 downsample)',
    build: () => {
      const c1 = new nn.Conv2d(2, 4, 3, { stride: 2, padding: 1 }), b1 = new nn.BatchNorm2d(4);
      const c2 = new nn.Conv2d(4, 4, 3, { padding: 1 }), b2 = new nn.BatchNorm2d(4);
      const dc = new nn.Conv2d(2, 4, 1, { stride: 2 }), db = new nn.BatchNorm2d(4);
      evalMod(b1, b2, db);
      const fl = new nn.Flatten(), fc = new nn.Linear(4 * 2 * 2, 3);
      return {
        fwd: (x) => {
          const idn = db.forward(dc.forward(x));
          const y = b2.forward(c2.forward(relu(b1.forward(c1.forward(x)))));
          return fc.forward(fl.forward(relu(T.add(y, idn))));
        },
        shapes: [[1, 2, 4, 4]],
      };
    },
  },
  {
    name: 'depthwise-separable conv (grouped)',
    build: () => {
      const dw = new nn.Conv2d(4, 4, 3, { padding: 1, groups: 4 }), pw = new nn.Conv2d(4, 6, 1);
      const gap = new nn.AdaptiveAvgPool2d(1), fl = new nn.Flatten(), fc = new nn.Linear(6, 3);
      return { fwd: (x) => fc.forward(fl.forward(gap.forward(relu(pw.forward(dw.forward(x)))))), shapes: [[1, 4, 4, 4]] };
    },
  },
  {
    name: 'Transformer encoder layer (attention + LN + gelu FFN + residual)',
    build: () => {
      const layer = new nn.TransformerEncoderLayer(8, 2, 16, 0.0, 'gelu', 1e-5, true, false);
      evalMod(layer);
      return { fwd: (x) => layer.forward(x), shapes: [[1, 3, 8]] };
    },
  },
  {
    name: 'pre-norm causal decoder block + lm head',
    build: () => {
      const blk = new nn.TransformerEncoderLayer(8, 2, 16, 0.0, 'gelu', 1e-5, true, true);
      evalMod(blk);
      const lnf = new nn.LayerNorm(8, 1e-5), head = new nn.Linear(8, 5);
      return { fwd: (x) => head.forward(lnf.forward(blk.forward(x, null, null, true))), shapes: [[1, 3, 8]] };
    },
  },
  {
    name: 'ViT patch-embed + transformer + mean pool',
    build: () => {
      const patch = new nn.Conv2d(2, 8, 2, { stride: 2 });
      const blk = new nn.TransformerEncoderLayer(8, 2, 16, 0.0, 'gelu', 1e-5, true, true);
      evalMod(blk);
      const lnf = new nn.LayerNorm(8, 1e-5), head = new nn.Linear(8, 3);
      return {
        fwd: (x) => {
          const e = patch.forward(x);
          const h = T.transpose(T.reshape(e, [e.shape[0], 8, 4]), 1, 2);
          return head.forward(T.mean(lnf.forward(blk.forward(h)), 1));
        },
        shapes: [[1, 2, 4, 4]],
      };
    },
  },
  {
    name: 'LSTM seq2seq + dot attention (explicit initial state)',
    build: () => {
      const enc = new nn.LSTM(4, 6, 1, true), dec = new nn.LSTM(4, 6, 1, true);
      const out = new nn.Linear(12, 5);
      return {
        fwd: (src, tgt, he, ce, hd, cd) => {
          const eo = enc.forward(src, [he, ce])[0];
          const dout = dec.forward(tgt, [hd, cd])[0];
          const ctx = T.matmul(T.softmax(T.matmul(dout, T.transpose(eo, 1, 2)), 2), eo);
          return out.forward(T.cat([dout, ctx], 2));
        },
        shapes: [[1, 3, 4], [1, 2, 4], [1, 1, 6], [1, 1, 6], [1, 1, 6], [1, 1, 6]],
      };
    },
  },
  {
    name: 'BiLSTM forward + reversed, concat (explicit initial state)',
    build: () => {
      const f = new nn.LSTM(4, 6, 1, true), b = new nn.LSTM(4, 6, 1, true), fc = new nn.Linear(12, 3);
      return {
        fwd: (x, hf0, cf0, hb0, cb0) => {
          const hf = f.forward(x, [hf0, cf0])[0];
          const hb = T.flip(b.forward(T.flip(x, [1]), [hb0, cb0])[0], [1]);
          return fc.forward(T.mean(T.cat([hf, hb], 2), 1));
        },
        shapes: [[1, 3, 4], [1, 1, 6], [1, 1, 6], [1, 1, 6], [1, 1, 6]],
      };
    },
  },
  {
    name: 'GRU stack (explicit initial state)',
    build: () => {
      const g = new nn.GRU(4, 6, 2, true), head = new nn.Linear(6, 3);
      return { fwd: (x, h0) => head.forward(g.forward(x, h0)[0]), shapes: [[1, 3, 4], [2, 1, 6]] };
    },
  },
  {
    name: 'U-Net (skip concat + nearest upsample)',
    build: () => {
      const c1 = new nn.Conv2d(1, 4, 3, { padding: 1 }), bott = new nn.Conv2d(4, 8, 3, { padding: 1 });
      const u1 = new nn.Conv2d(8 + 4, 4, 3, { padding: 1 }), head = new nn.Conv2d(4, 2, 1);
      const p = new nn.MaxPool2d(2);
      return {
        fwd: (x) => {
          const s1 = relu(c1.forward(x));
          const b = relu(bott.forward(p.forward(s1)));
          return head.forward(relu(u1.forward(T.cat([upsample2x(b), s1], 1))));
        },
        shapes: [[1, 1, 4, 4]],
      };
    },
  },
  {
    name: 'TextCNN (conv1d + max over time)',
    build: () => {
      const convs = [2, 3].map((k) => new nn.Conv1d(4, 3, k));
      const fc = new nn.Linear(6, 3);
      return {
        fwd: (x) => fc.forward(T.cat(convs.map((c) => T.max(relu(c.forward(x)), 2)), 1)),
        shapes: [[1, 4, 6]],
      };
    },
  },
  {
    name: 'two-tower (shared-shape encoders + dot score)',
    build: () => {
      const ea = [new nn.Linear(5, 6), new nn.Linear(6, 4)];
      const eb = [new nn.Linear(5, 6), new nn.Linear(6, 4)];
      const run = (enc, x) => enc[1].forward(relu(enc[0].forward(x)));
      return { fwd: (a, b) => T.matmul(run(ea, a), T.transpose(run(eb, b), 0, 1)), shapes: [[2, 5], [2, 5]] };
    },
  },
];

function writeInto(t, values) {
  const raw = t._impl.storage.data;
  for (let i = 0; i < values.length; i++) raw[i] = values[i];
  t._impl.bumpVersion();
}

function probeIndices(n) {
  const step = Math.max(1, Math.floor(n / PROBES_PER_TENSOR));
  const out = [];
  for (let k = 0; k < n && out.length < PROBES_PER_TENSOR; k += step) out.push(k);
  return out;
}

describe('model-level VJP matches finite differences (inputs and parameters)', () => {
  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 6577 + 13);
      const { fwd, shapes } = c.build(rng);
      const datas = shapes.map((s) => randomNested(rng, s));
      const inputs = datas.map((d) => tensor(d));

      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget() });
      const out = cf(...inputs);
      const grads = cf.backward(ones(out.shape));
      const params = cf.capturedParams();

      expect(grads.length, 'backward must return one gradient per input then per captured parameter')
        .toBe(inputs.length + params.length);

      const lossAt = () => flat(fwd(...datas.map((d) => tensor(d)))).reduce((a, b) => a + b, 0);

      const checkTensor = (label, analytic, n, read, write) => {
        expect(analytic.length, `${label}: gradient size`).toBe(n);
        const base = Array.from(read());
        for (const k of probeIndices(n)) {
          const at = (delta) => {
            const arr = Array.from(base);
            arr[k] += delta;
            write(arr);
            const v = lossAt();
            write(base);
            return v;
          };
          const numeric = (at(EPS) - at(-EPS)) / (2 * EPS);
          const err = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
          expect(err, `${label}[${k}]: numeric=${numeric} analytic=${analytic[k]}`).toBeLessThan(TOL);
        }
      };

      for (let i = 0; i < inputs.length; i++) {
        const n = numel(shapes[i]);
        checkTensor(
          `input${i}`,
          flat(grads[i]),
          n,
          () => flat(tensor(datas[i])),
          (arr) => { datas[i] = nest(arr, shapes[i]); },
        );
      }

      const trainable = params.map((p, j) => ({ p, j })).filter(({ p }) => p.isParameter === true);
      expect(trainable.length, 'the traced graph must capture the module parameters').toBeGreaterThan(0);

      const stride = Math.max(1, Math.ceil(trainable.length / MAX_PARAM_TENSORS));
      let checked = 0;
      for (let t = 0; t < trainable.length; t += stride) {
        const { p, j } = trainable[t];
        checkTensor(
          `param${j}[${p.shape}]`,
          flat(grads[inputs.length + j]),
          numel([...p.shape]),
          () => flat(p),
          (arr) => writeInto(p, arr),
        );
        checked++;
      }
      expect(checked, 'at least one parameter gradient must be checked').toBeGreaterThan(0);
    });
  }
});
