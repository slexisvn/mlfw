import { describe, it, expect, afterAll } from 'vitest';
import { tensor, compileWithBackward, CPUTarget, CUDATarget } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import * as T from '../../../src/tensor/ops/ops.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { releaseCudaMemory } from '../../../src/runtime/backend_registry.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat } from '../../_utils/tensor_data.js';

const TOL = 5e-3;
const relu = nn.F.relu;

function evalMod(...ms) { ms.forEach((m) => m.eval()); return ms; }

afterAll(async () => { await releaseCudaMemory(); });

const CASES = [
  {
    name: 'MLP (linear + gelu)',
    build: () => {
      const a = new nn.Linear(8, 12), b = new nn.Linear(12, 4);
      return { fwd: (x) => b.forward(nn.F.gelu(a.forward(x))), shapes: [[2, 8]] };
    },
  },
  {
    name: 'CNN (conv + batchnorm + relu + pool + linear)',
    build: () => {
      const c1 = new nn.Conv2d(2, 4, 3, { padding: 1 }), b1 = new nn.BatchNorm2d(4);
      const p = new nn.MaxPool2d(2), fl = new nn.Flatten(), fc = new nn.Linear(4 * 4 * 4, 3);
      evalMod(b1);
      return { fwd: (x) => fc.forward(fl.forward(p.forward(relu(b1.forward(c1.forward(x)))))), shapes: [[2, 2, 8, 8]] };
    },
  },
  {
    name: 'ResNet block (residual add + 1x1 downsample)',
    build: () => {
      const c1 = new nn.Conv2d(2, 4, 3, { stride: 2, padding: 1 }), b1 = new nn.BatchNorm2d(4);
      const c2 = new nn.Conv2d(4, 4, 3, { padding: 1 }), b2 = new nn.BatchNorm2d(4);
      const dc = new nn.Conv2d(2, 4, 1, { stride: 2 }), db = new nn.BatchNorm2d(4);
      evalMod(b1, b2, db);
      const fl = new nn.Flatten(), fc = new nn.Linear(4 * 4 * 4, 3);
      return {
        fwd: (x) => {
          const idn = db.forward(dc.forward(x));
          const y = b2.forward(c2.forward(relu(b1.forward(c1.forward(x)))));
          return fc.forward(fl.forward(relu(T.add(y, idn))));
        },
        shapes: [[2, 2, 8, 8]],
      };
    },
  },
  {
    name: 'transformer encoder layer',
    build: () => {
      const layer = new nn.TransformerEncoderLayer(8, 2, 16, 0.0, 'gelu', 1e-5, true, false);
      evalMod(layer);
      return { fwd: (x) => layer.forward(x), shapes: [[2, 4, 8]] };
    },
  },
  {
    name: 'pre-norm causal decoder block + lm head',
    build: () => {
      const blk = new nn.TransformerEncoderLayer(8, 2, 16, 0.0, 'gelu', 1e-5, true, true);
      evalMod(blk);
      const lnf = new nn.LayerNorm(8, 1e-5), head = new nn.Linear(8, 5);
      return { fwd: (x) => head.forward(lnf.forward(blk.forward(x, null, null, true))), shapes: [[2, 4, 8]] };
    },
  },
  {
    name: 'GRU stack (explicit initial state)',
    build: () => {
      const g = new nn.GRU(4, 6, 2, true), head = new nn.Linear(6, 3);
      return { fwd: (x, h0) => head.forward(g.forward(x, h0)[0]), shapes: [[2, 3, 4], [2, 2, 6]] };
    },
  },
  {
    name: 'LSTM + dot attention (explicit initial state)',
    build: () => {
      const enc = new nn.LSTM(4, 6, 1, true), out = new nn.Linear(6, 5);
      return {
        fwd: (src, h, c) => {
          const eo = enc.forward(src, [h, c])[0];
          const ctx = T.matmul(T.softmax(T.matmul(eo, T.transpose(eo, 1, 2)), 2), eo);
          return out.forward(ctx);
        },
        shapes: [[2, 3, 4], [1, 2, 6], [1, 2, 6]],
      };
    },
  },
  {
    name: 'depthwise-separable conv (grouped)',
    build: () => {
      const dw = new nn.Conv2d(4, 4, 3, { padding: 1, groups: 4 }), pw = new nn.Conv2d(4, 6, 1);
      const gap = new nn.AdaptiveAvgPool2d(1), fl = new nn.Flatten(), fc = new nn.Linear(6, 3);
      return { fwd: (x) => fc.forward(fl.forward(gap.forward(relu(pw.forward(dw.forward(x)))))), shapes: [[2, 4, 8, 8]] };
    },
  },
];

async function gradsOn(target, fwd, datas) {
  const inputs = datas.map((d) => tensor(d));
  const cf = compileWithBackward({ forward: fwd }, inputs, { target, mode: 'separate' });
  const out = await cf(...inputs);
  const grads = await cf.backward(ones(out.shape));
  return { grads: grads.map((g) => (g == null ? null : flat(g))), out: flat(out) };
}

describe.each(['separate'])('compiled backward on CUDA matches CPU (%s mode)', () => {
  for (const c of CASES) {
    it(c.name, async () => {
      const rng = mulberry32(c.name.length * 3121 + 7);
      const { fwd, shapes } = c.build();
      const datas = shapes.map((s) => randomNested(rng, s));

      const cpu = await gradsOn(CPUTarget(), fwd, datas);
      const gpu = await gradsOn(CUDATarget(), fwd, datas);

      expect(gpu.out.length, 'forward output size').toBe(cpu.out.length);
      expect(gpu.grads.length, 'gradient count').toBe(cpu.grads.length);

      for (let i = 0; i < cpu.grads.length; i++) {
        if (cpu.grads[i] === null) { expect(gpu.grads[i]).toBeNull(); continue; }
        expect(gpu.grads[i], `grad ${i} missing on cuda`).not.toBeNull();
        expect(gpu.grads[i].length, `grad ${i} size`).toBe(cpu.grads[i].length);

        let maxErr = 0, bad = -1;
        for (let k = 0; k < cpu.grads[i].length; k++) {
          expect(Number.isFinite(gpu.grads[i][k]), `grad ${i}[${k}] is not finite on cuda`).toBe(true);
          const e = Math.abs(cpu.grads[i][k] - gpu.grads[i][k]) / (1 + Math.abs(cpu.grads[i][k]));
          if (e > maxErr) { maxErr = e; bad = k; }
        }
        expect(maxErr, `grad ${i}[${bad}]: cpu=${cpu.grads[i][bad]} cuda=${gpu.grads[i][bad]}`).toBeLessThan(TOL);
      }
    });
  }
});
