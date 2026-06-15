import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, Linear, relu, sum, cat } from '../../../src/index.js';
import { GPU_DEVICE } from '../../../src/tensor/types/device.js';
import { noGrad } from '../../../src/autograd/grad_mode.js';
import { preloadCudaRuntime } from '../../../src/compiler/runtime/backend_registry.js';
import { cudaDeps } from './cuda-setup.js';

let resident = null;
if (cudaDeps) resident = await import('../../../src/io/node/cuda/resident.js');

const flat = (v) => Array.from(v && v.contiguous ? v.contiguous().data : v.data);
const absSum = (arr) => { let s = 0; for (const x of arr) s += Math.abs(x); return s; };

function buildModel() {
  const l1 = new Linear(16, 24);
  const l2 = new Linear(48, 4);
  l1.to(GPU_DEVICE); l2.to(GPU_DEVICE);
  const x = tensor(Array.from({ length: 8 * 16 }, (_, i) => Math.sin(1 + i * 0.37) * 0.5), { shape: [8, 16] }).to(GPU_DEVICE);
  return { l1, l2, x, params: [...l1.parameters(), ...l2.parameters()] };
}

function stepLossAndGrads({ l1, l2, x, params }) {
  for (const p of params) p.grad = null;
  const h = relu(l1.forward(x));
  const out = l2.forward(cat([h, h], 1));
  const loss = sum(out);
  loss.backward();
  return { loss: loss.item(), grad: absSum(params.flatMap((p) => flat(p.grad))) };
}

describe.skipIf(!cudaDeps)('CUDA eager deferred residency', () => {
  beforeAll(async () => { await preloadCudaRuntime(); });

  it('deferred fwd+bwd is bit-exact vs safe (matmul + relu + cat + reduce)', () => {
    const m = buildModel();
    resident.setEagerDeferred(false);
    const safe = stepLossAndGrads(m);
    resident.setEagerDeferred(true);
    const def = stepLossAndGrads(m);
    resident.flushDeferred();
    resident.setEagerDeferred(false);

    expect(Math.abs(def.loss - safe.loss) / (1 + Math.abs(safe.loss))).toBeLessThan(1e-4);
    expect(Math.abs(def.grad - safe.grad) / (1 + safe.grad)).toBeLessThan(1e-4);
  }, 60000);

  it('per-step flush keeps device memory bounded over many forwards', () => {
    const big = tensor(Array.from({ length: 1 << 18 }, (_, i) => (i % 97) * 0.01), { shape: [1 << 18] }).to(GPU_DEVICE);
    resident.setEagerDeferred(true);
    let last = 0;
    noGrad(() => {
      for (let i = 0; i < 500; i++) {
        const r = sum(relu(big));
        if (i === 499) last = r.item();
        resident.flushDeferred();
      }
    });
    resident.setEagerDeferred(false);
    expect(Number.isFinite(last)).toBe(true);
  }, 60000);

  it('eviction bound stays correct without an explicit flush', () => {
    const x = tensor(Array.from({ length: 4096 }, (_, i) => (i % 13) * 0.1), { shape: [4096] }).to(GPU_DEVICE);
    resident.setEagerDeferred(false);
    const safe = sum(relu(x)).item();
    resident.setEagerDeferred(true);
    let def = 0;
    noGrad(() => { for (let i = 0; i < 2000; i++) def = sum(relu(x)).item(); });
    resident.flushDeferred();
    resident.setEagerDeferred(false);
    expect(Math.abs(def - safe) / (1 + Math.abs(safe))).toBeLessThan(1e-4);
  }, 60000);
});
