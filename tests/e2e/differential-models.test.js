import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { compile } from '../../src/tracing/compile.js';
import { CPUTarget, WasmTarget, CUDATarget } from '../../src/backend/target.js';
import { buildGpt2 } from '../_utils/gpt2_model.js';
import { mulberry32 } from '../_utils/rng.js';
import { numel, nest, randomTensor, flat } from '../_utils/tensor_data.js';

const EXEC = [['cpu', CPUTarget], ['wasm', WasmTarget]];
const CODEGEN = [['gpu', CUDATarget]];

async function expectMatchesEager(fwd, inputs, tol = 2e-3) {
  const eager = flat(fwd(...inputs));
  for (const [name, makeTarget] of EXEC) {
    const compiled = compile({ forward: fwd }, inputs, { target: makeTarget() });
    const out = flat(await compiled(...inputs));
    expect(out.length, `${name} length`).toBe(eager.length);
    let maxErr = 0, bad = -1;
    for (let i = 0; i < eager.length; i++) {
      const e = Math.abs(eager[i] - out[i]) / (1 + Math.abs(eager[i]));
      if (e > maxErr) { maxErr = e; bad = i; }
    }
    expect(maxErr, `${name} idx ${bad}: eager=${eager[bad]} compiled=${out[bad]}`).toBeLessThan(tol);
  }
  for (const [name, makeTarget] of CODEGEN) {
    const compiled = compile({ forward: fwd }, inputs, { target: makeTarget() });
    const src = compiled.source();
    expect(compiled.kernels().length, `${name} emitted no kernel`).toBeGreaterThan(0);
    expect(src, `${name} source is not CUDA`).toMatch(/__global__\s+void/);
    expect(src, `${name} leaked a JS artifact into CUDA`).not.toMatch(/\bMath\.\w+|\bnew Float32Array\b|=>/);
  }
}

describe('large models end-to-end (compiled == eager, all backends)', () => {
  it('Transformer encoder stack (attention + layernorm + gelu FFN + residual)', async () => {
    const rng = mulberry32(11);
    const D = 16, H = 2, FF = 32, L = 2, SEQ = 4, B = 1;
    const layers = Array.from({ length: L }, () => new nn.TransformerEncoderLayer(D, H, FF, 0.0, 'gelu', 1e-5, true, false));
    layers.forEach((m) => m.eval());
    const fwd = (x) => { let h = x; for (const layer of layers) h = layer.forward(h); return h; };
    await expectMatchesEager(fwd, [randomTensor(rng, [B, SEQ, D])]);
  });

  it('CNN classifier (conv + batchnorm + relu + maxpool + linear)', async () => {
    const rng = mulberry32(22);
    const conv1 = new nn.Conv2d(3, 4, 3, { stride: 1, padding: 1 }), bn1 = new nn.BatchNorm2d(4);
    const conv2 = new nn.Conv2d(4, 8, 3, { stride: 1, padding: 1 }), bn2 = new nn.BatchNorm2d(8);
    const pool = new nn.MaxPool2d(2), flatten = new nn.Flatten(), fc = new nn.Linear(8 * 4 * 4, 10);
    [bn1, bn2].forEach((m) => m.eval());
    const relu = nn.F.relu;
    const fwd = (x) => fc.forward(flatten.forward(pool.forward(relu(bn2.forward(conv2.forward(relu(bn1.forward(conv1.forward(x)))))))));
    await expectMatchesEager(fwd, [randomTensor(rng, [1, 3, 8, 8])]);
  });

  it('BatchNorm in training mode (batch statistics) compiles == eager', async () => {
    const rng = mulberry32(44);
    const bn2d = new nn.BatchNorm2d(8);
    expect(bn2d.training).toBe(true);
    await expectMatchesEager((x) => bn2d.forward(x), [randomTensor(rng, [4, 8, 6, 6])]);
    const bn1d = new nn.BatchNorm1d(16);
    await expectMatchesEager((x) => bn1d.forward(x), [randomTensor(rng, [10, 16])]);
  });

  it('deep MLP (6 linear layers + gelu)', async () => {
    const rng = mulberry32(33);
    const dims = [32, 48, 48, 48, 48, 48, 10];
    const lins = dims.slice(0, -1).map((d, i) => new nn.Linear(d, dims[i + 1]));
    const fwd = (x) => { let h = x; for (let i = 0; i < lins.length; i++) { h = lins[i].forward(h); if (i < lins.length - 1) h = nn.F.gelu(h); } return h; };
    await expectMatchesEager(fwd, [randomTensor(rng, [4, 32])]);
  });

  it('GPT-2 decoder (token+pos embedding + causal MHA + pre-norm + gelu FFN + lm head)', async () => {
    const { fwd, ids } = buildGpt2(nn, tensor, { V: 128, D: 64, H: 4, FF: 256, L: 3, SEQ: 12 });
    await expectMatchesEager(fwd, [ids]);
  });
});
