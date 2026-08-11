import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, Adam, Linear, LSTM, relu, MSELoss, TensorDataset, DataLoader } from '../../../src/index.js';
import { LightningModule, Trainer } from '../../../src/lightning/index.js';
import { GradMode } from '../../../src/autograd/grad_mode.js';
import { GPU_DEVICE } from '../../../src/tensor/types/device.js';
import { preloadCudaRuntime } from '../../../src/runtime/backend_registry.js';
import { cudaDeps } from '../../_utils/cuda.js';
import { getDevice } from '../../../src/runtime/node/cuda/device.js';
import { acquire, release, drainPool } from '../../../src/runtime/node/cuda/memory.js';
import { releaseCudaMemory } from '../../../src/runtime/node/cuda_runtime.js';
import { cu, checkCU } from '../../../src/runtime/node/cuda/ffi.js';

function freeBytes() {
  const free = [0n], total = [0n];
  checkCU('cuMemGetInfo', cu.memGetInfo(free, total));
  return Number(free[0]);
}

const MB = 1024 * 1024;
const quiet = { logger: false, enableProgress: false, enableCheckpointing: false };

describe.skipIf(!cudaDeps)('CUDA memory is released, not held until process exit', () => {
  beforeAll(async () => { await preloadCudaRuntime(); getDevice(); });

  it('the caching pool holds freed blocks (the symptom); drainPool returns them to the driver (the fix)', () => {
    const base = freeBytes();
    const N = 16, SZ = 4 * MB;
    const ptrs = [];
    for (let i = 0; i < N; i++) ptrs.push(acquire(SZ));
    expect(base - freeBytes(), 'allocation should drop free device memory').toBeGreaterThan(N * SZ * 0.5);

    for (const p of ptrs) release(p, SZ);
    expect(base - freeBytes(), 'release() only recycles into the pool — memory is NOT returned to the driver').toBeGreaterThan(N * SZ * 0.5);

    const freed = drainPool();
    expect(freed).toBeGreaterThanOrEqual(N);
    expect(freeBytes(), 'after drainPool the device memory is reclaimed (~baseline)').toBeGreaterThanOrEqual(base - 8 * MB);
  });

  it('eval-mode LSTM generation on GPU does not leak (cuDNN inference path releases reserve/work)', () => {
    const m = new LSTM(32, 128, 3, true);
    m.to(GPU_DEVICE);
    m.eval();
    const x = tensor(Array.from({ length: 1 }, () => Array.from({ length: 1 }, () => Array.from({ length: 32 }, (_, k) => 0.01 * k)))).to(GPU_DEVICE);
    const prev = GradMode.isEnabled();
    GradMode.setEnabled(false);
    try {
      m.forward(x)[0].toArray();
      const start = freeBytes();
      for (let i = 0; i < 200; i++) m.forward(x)[0].toArray();
      const leaked = start - freeBytes();
      expect(leaked, `eval LSTM leaked ${Math.round(leaked / MB)}MB over 200 forwards`).toBeLessThan(16 * MB);
    } finally {
      GradMode.setEnabled(prev);
    }
  });

  it('releaseCudaMemory() reclaims pooled + resident device memory', () => {
    const base = freeBytes();
    const ptrs = [];
    for (let i = 0; i < 12; i++) ptrs.push(acquire(4 * MB));
    for (const p of ptrs) release(p, 4 * MB);
    expect(base - freeBytes()).toBeGreaterThan(24 * MB);
    releaseCudaMemory();
    expect(freeBytes()).toBeGreaterThanOrEqual(base - 8 * MB);
  });
});

class RegMLP extends LightningModule {
  constructor(inF) { super(); this.l1 = new Linear(inF, 256); this.l2 = new Linear(256, inF); this.lossFn = new MSELoss(); }
  forward(x) { return this.l2.forward(relu(this.l1.forward(x))); }
  trainingStep(batch) { const [x, y] = batch; return this.lossFn.forward(this.forward(x), y); }
  configureOptimizers() { return new Adam(this.parameters(), { lr: 0.01 }); }
}

function makeLoader(inF, n, b) {
  const xf = []; for (let i = 0; i < n * inF; i++) xf.push(Math.sin(i * 0.13));
  const yf = []; for (let i = 0; i < n * inF; i++) yf.push(Math.cos(i * 0.17));
  const nest = (f, cols) => { const o = []; for (let r = 0; r < f.length / cols; r++) o.push(f.slice(r * cols, (r + 1) * cols)); return o; };
  return new DataLoader(new TensorDataset(tensor(nest(xf, inF)), tensor(nest(yf, inF))), { batchSize: b });
}

describe.skipIf(!cudaDeps)('Trainer(accelerator="gpu").fit releases device memory and leaves the model usable', () => {
  beforeAll(async () => { await preloadCudaRuntime(); getDevice(); });

  it('after fit, the trained model still infers correctly (params re-upload from preserved host weights)', async () => {
    const base = freeBytes();
    const m = new RegMLP(64);
    await new Trainer({ accelerator: 'gpu', maxEpochs: 3, ...quiet }).fit(m, makeLoader(64, 256, 32));

    expect(base - freeBytes(), 'most device memory reclaimed after fit teardown').toBeLessThan(96 * MB);

    const x = tensor(Array.from({ length: 4 }, (_, r) => Array.from({ length: 64 }, (_, c) => Math.sin((r * 64 + c) * 0.07))));
    m.eval();
    const g1 = m.forward(x.to(GPU_DEVICE)).toArray().flat(Infinity);
    const g2 = m.forward(x.to(GPU_DEVICE)).toArray().flat(Infinity);
    expect(g1.every(Number.isFinite), 'forward after teardown produces finite output').toBe(true);
    expect(g1.some((v) => v !== 0), 'output is not collapsed to zero (weights intact)').toBe(true);
    expect(g2, 'forward is deterministic after teardown (re-upload works, no corruption)').toEqual(g1);
  });
});
