import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, sum, add } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { GPU_DEVICE } from '../../../src/tensor/types/device.js';
import { getCudnnGRU } from '../../../src/dispatcher/jit_dispatch.js';
import { preloadCudaRuntime } from '../../../src/runtime/backend_registry.js';
import { cudaDeps } from './cuda-setup.js';

const haveCudnn = cudaDeps && !!getCudnnGRU();
const flat = (t) => Array.from(t.contiguous().data);

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const arr = (rng, n) => Array.from({ length: n }, () => rng() * 2 - 1);

function gradsOf(m) {
  return [...m.parameters()].map((p) => flat(p.grad));
}
function relMax(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return m;
}

describe.skipIf(!haveCudnn)('cuDNN GRU: nn.GRU on GPU matches CPU elementwise (fwd + bwd)', () => {
  beforeAll(async () => { await preloadCudaRuntime(); });
  const configs = [
    { I: 4, H: 6, L: 1, T: 5, B: 3, bf: false },
    { I: 4, H: 6, L: 2, T: 5, B: 3, bf: false },
    { I: 5, H: 8, L: 3, T: 4, B: 2, bf: true },
  ];

  for (const cfg of configs) {
    const { I, H, L, T, B, bf } = cfg;
    it(`L=${L} H=${H} batchFirst=${bf} fwd+bwd`, () => {
      const rng = mulberry32(1234 + L * 7 + H);
      const xData = arr(rng, T * B * I);
      const shape = bf ? [B, T, I] : [T, B, I];
      const gru = new nn.GRU(I, H, L, bf);

      const xc = tensor(xData, { shape, requiresGrad: true });
      const [oc, hc] = gru.forward(xc);
      sum(add(sum(oc), sum(hc))).backward();
      const refOut = flat(oc), refHn = flat(hc), refDx = flat(xc.grad), refG = gradsOf(gru);

      for (const p of gru.parameters()) p.grad = null;
      gru.to(GPU_DEVICE);
      const xg = tensor(xData, { shape }).to(GPU_DEVICE); xg.requiresGrad_(true);
      const [og, hg] = gru.forward(xg);
      sum(add(sum(og), sum(hg))).backward();

      expect(relMax(refOut, flat(og)), 'output').toBeLessThan(1e-2);
      expect(relMax(refHn, flat(hg)), 'final h').toBeLessThan(1e-2);
      expect(relMax(refDx, flat(xg.grad)), 'input grad').toBeLessThan(1e-2);
      const gpuG = gradsOf(gru);
      for (let i = 0; i < refG.length; i++) expect(relMax(refG[i], gpuG[i]), `param ${i} grad`).toBeLessThan(1e-2);
    }, 60000);
  }

  it('initial state + final-state grads', () => {
    const I = 4, H = 6, L = 2, T = 5, B = 3;
    const rng = mulberry32(99);
    const xData = arr(rng, T * B * I), hData = arr(rng, L * B * H);
    const gru = new nn.GRU(I, H, L, false);

    const ref = () => {
      const x = tensor(xData, { shape: [T, B, I], requiresGrad: true });
      const h0 = tensor(hData, { shape: [L, B, H], requiresGrad: true });
      const [out, hn] = gru.forward(x, h0);
      sum(add(sum(out), sum(hn))).backward();
      return { out: flat(out), hn: flat(hn), dx: flat(x.grad), dh: flat(h0.grad), g: gradsOf(gru) };
    };
    const r = ref();

    for (const p of gru.parameters()) p.grad = null;
    gru.to(GPU_DEVICE);
    const x = tensor(xData, { shape: [T, B, I] }).to(GPU_DEVICE); x.requiresGrad_(true);
    const h0 = tensor(hData, { shape: [L, B, H] }).to(GPU_DEVICE); h0.requiresGrad_(true);
    const [out, hn] = gru.forward(x, h0);
    sum(add(sum(out), sum(hn))).backward();

    expect(relMax(r.out, flat(out)), 'output').toBeLessThan(1e-2);
    expect(relMax(r.hn, flat(hn)), 'final h').toBeLessThan(1e-2);
    expect(relMax(r.dx, flat(x.grad)), 'input grad').toBeLessThan(1e-2);
    expect(relMax(r.dh, flat(h0.grad)), 'init-h grad').toBeLessThan(1e-2);
    const g = gradsOf(gru);
    for (let i = 0; i < r.g.length; i++) expect(relMax(r.g[i], g[i]), `param ${i} grad`).toBeLessThan(1e-2);
  }, 60000);
});
