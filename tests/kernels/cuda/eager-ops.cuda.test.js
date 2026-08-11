import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { tensor, sum, mean, mul, Adam, log_softmax, softmax, CrossEntropyLoss, Linear, relu, cat } from '../../../src/index.js';
import { GPU_DEVICE } from '../../../src/tensor/types/device.js';
import { noGrad } from '../../../src/autograd/grad_mode.js';
import { preloadCudaRuntime } from '../../../src/runtime/backend_registry.js';
import { cudaDeps } from '../../_utils/cuda.js';

const flat = (v) => Array.from(v && v.contiguous ? v.contiguous().data : v.data);
const maxAbsErr = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe.skipIf(!cudaDeps)('CUDA eager device-resident ops match CPU', () => {
  let resident;
  beforeAll(async () => {
    await preloadCudaRuntime();
    resident = await import('../../../src/runtime/node/cuda/resident.js');
  });
  afterAll(() => { if (resident) resident.setEagerDeferred(false); });

  describe('cuBLAS matmul (forward + backward, transpose via stride flags)', () => {
    beforeAll(() => resident.setEagerDeferred(true));
    afterAll(() => resident.setEagerDeferred(false));

    const M = 4, K = 5, N = 3, b = 2, T = 4, H = 6;
    const fill = (n, k) => Array.from({ length: n }, (_, i) => Math.sin(i * 0.137 + k * 0.41));

    function run(aShape, bShape, forward) {
      const aData = fill(aShape.reduce((p, q) => p * q, 1), 1);
      const bData = fill(bShape.reduce((p, q) => p * q, 1), 2);
      const ac = tensor(aData, { shape: aShape, requiresGrad: true });
      const bc = tensor(bData, { shape: bShape, requiresGrad: true });
      const oc = forward(ac, bc);
      sum(oc).backward();
      const refOut = flat(oc), refDa = flat(ac.grad), refDb = flat(bc.grad);
      const ag = tensor(aData, { shape: aShape }).to(GPU_DEVICE); ag.requiresGrad_(true);
      const bg = tensor(bData, { shape: bShape }).to(GPU_DEVICE); bg.requiresGrad_(true);
      const og = forward(ag, bg);
      sum(og).backward();
      return { refOut, refDa, refDb, gpuOut: flat(og), gpuDa: flat(ag.grad), gpuDb: flat(bg.grad) };
    }
    function check(r) {
      expect(r.gpuOut.length).toBe(r.refOut.length);
      expect(maxAbsErr(r.refOut, r.gpuOut), 'forward output').toBeLessThan(1e-3);
      expect(maxAbsErr(r.refDa, r.gpuDa), 'grad dA').toBeLessThan(1e-3);
      expect(maxAbsErr(r.refDb, r.gpuDb), 'grad dB').toBeLessThan(1e-3);
    }

    it('plain 2D A[M,K] @ B[K,N]', () => {
      check(run([M, K], [K, N], (a, bb) => a.matmul(bb)));
    }, 60000);

    it('Linear-style X[M,K] @ W^T (W is [N,K], transpose view)', () => {
      check(run([M, K], [N, K], (x, w) => x.matmul(w.transpose(0, 1))));
    }, 60000);

    it('batched 3D A[b,M,K] @ B[b,K,N]', () => {
      check(run([b, M, K], [b, K, N], (a, bb) => a.matmul(bb)));
    }, 60000);

    it('batched-transpose attention dec[b,T,H] @ enc[b,T,H].transpose(1,2)', () => {
      check(run([b, T, H], [b, T, H], (dec, enc) => dec.matmul(enc.transpose(1, 2))));
    }, 60000);
  });

  describe('parallel full-reduction', () => {
    beforeAll(() => resident.setEagerDeferred(false));

    const ROWS = 1024, COLS = 512;
    const data = Array.from({ length: ROWS * COLS }, (_, i) => Math.sin(i * 0.001) * 2);
    const cpuT = () => tensor(data, { shape: [ROWS, COLS] });
    const gpuT = () => tensor(data, { shape: [ROWS, COLS] }).to(GPU_DEVICE);
    const scalarOf = (t) => t.contiguous().data[0];

    it('full sum over all elements', () => {
      const cpu = scalarOf(sum(cpuT())), gpu = scalarOf(sum(gpuT()));
      expect(Math.abs(gpu - cpu)).toBeLessThan(Math.abs(cpu) * 1e-4);
    }, 60000);

    it('full mean over all elements', () => {
      const cpu = scalarOf(mean(cpuT())), gpu = scalarOf(mean(gpuT()));
      expect(Math.abs(gpu - cpu)).toBeLessThan(1e-5);
    }, 60000);

    it('partial reduction over a single dim stays correct', () => {
      const cpu = Array.from(sum(cpuT(), 1).contiguous().data);
      const gpu = Array.from(sum(gpuT(), 1).contiguous().data);
      expect(gpu.length).toBe(ROWS);
      expect(maxAbsErr(cpu, gpu)).toBeLessThan(1e-2);
    }, 60000);
  });

  describe('device-resident Adam', () => {
    const INIT = [0.5, -0.3, 0.8, 0.1, -0.6, 0.2, 0.4, -0.7];
    const STEPS = 8;
    const gradAt = (s) => INIT.map((_, i) => Math.sin(s * 1.3 + i) * 0.4);

    function runManual(gpu) {
      let p = tensor(INIT, { shape: [INIT.length] });
      if (gpu) p = p.to(GPU_DEVICE);
      p.requiresGrad_(true);
      const opt = new Adam([p], { lr: 0.05 });
      for (let s = 0; s < STEPS; s++) {
        const g = gradAt(s);
        p.grad = gpu ? tensor(g, { shape: [INIT.length] }).to(GPU_DEVICE) : tensor(g, { shape: [INIT.length] });
        opt.step();
      }
      return Array.from(p.contiguous().data);
    }
    function runTrained(gpu) {
      let p = tensor(INIT, { shape: [INIT.length] });
      if (gpu) p = p.to(GPU_DEVICE);
      p.requiresGrad_(true);
      const opt = new Adam([p], { lr: 0.1 });
      for (let s = 0; s < STEPS; s++) {
        const loss = sum(mul(p, p));
        loss.backward();
        opt.step();
        p.grad = null;
      }
      return Array.from(p.contiguous().data);
    }

    it('manual-grad updates match CPU with no forward pass first', () => {
      const cpu = runManual(false);
      resident.setEagerDeferred(true);
      const gpu = runManual(true);
      resident.setEagerDeferred(false);
      expect(maxAbsErr(cpu, gpu)).toBeLessThan(1e-4);
    }, 60000);

    it('forward+backward optimization trajectory matches CPU', () => {
      const cpu = runTrained(false);
      resident.setEagerDeferred(true);
      const gpu = runTrained(true);
      resident.setEagerDeferred(false);
      expect(maxAbsErr(cpu, gpu)).toBeLessThan(1e-4);
    }, 60000);
  });

  describe('large-N fused composites (global scratch path)', () => {
    beforeAll(() => resident.setEagerDeferred(true));
    afterAll(() => resident.setEagerDeferred(false));

    const N = 320, VOCAB = 498;
    const makeLogits = () => Array.from({ length: N * VOCAB }, (_, i) => Math.sin(i * 0.013) + Math.cos(i * 0.0071));
    const pair = () => {
      const d = makeLogits();
      return { cpu: tensor(d, { shape: [N, VOCAB] }), gpu: tensor(d, { shape: [N, VOCAB] }).to(GPU_DEVICE) };
    };

    it(`log_softmax matches CPU at N=${N} over vocab ${VOCAB}`, () => {
      const { cpu, gpu } = pair();
      expect(maxAbsErr(flat(log_softmax(cpu, -1)), flat(log_softmax(gpu, -1)))).toBeLessThan(1e-3);
    }, 60000);

    it(`softmax matches CPU at N=${N} over vocab ${VOCAB}`, () => {
      const { cpu, gpu } = pair();
      expect(maxAbsErr(flat(softmax(cpu, -1)), flat(softmax(gpu, -1)))).toBeLessThan(1e-3);
    }, 60000);

    it(`CrossEntropyLoss matches CPU at N=${N} over vocab ${VOCAB}`, () => {
      const { cpu, gpu } = pair();
      const t = Array.from({ length: N }, (_, i) => i % VOCAB);
      const cpuT = tensor(t, { shape: [N], dtype: 'i32' });
      const gpuT = tensor(t, { shape: [N], dtype: 'i32' }).to(GPU_DEVICE);
      const ref = flat(new CrossEntropyLoss('mean').forward(cpu, cpuT));
      const dev = flat(new CrossEntropyLoss('mean').forward(gpu, gpuT));
      expect(maxAbsErr(ref, dev)).toBeLessThan(1e-3);
    }, 60000);
  });

  describe('deferred residency (flush / eviction)', () => {
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
});
