import { describe, it, expect, beforeAll } from 'vitest';
import { tensor, linalg, ml, matmul } from '../../src/index.js';
import { GPU_DEVICE } from '../../src/tensor/types/device.js';
import { preloadCudaRuntime } from '../../src/runtime/backend_registry.js';
import { cudaDeps } from '../backend/cuda/cuda-setup.js';

const flat = (t) => Array.from((t && t.contiguous ? t.contiguous() : t).data);
const maxAbsErr = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};
const relErr = (a, b) => {
  let m = 0, s = 0;
  for (let i = 0; i < a.length; i++) { m = Math.max(m, Math.abs(a[i] - b[i])); s = Math.max(s, Math.abs(a[i])); }
  return m / (s || 1);
};

const TOL = 2e-3;

function det2(seed, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[i * cols + j] = Math.sin(i * 0.7 + j * 1.3 + seed) + 0.05 * (i - j);
  return out;
}

function symmetric(n, seed) {
  const m = det2(seed, n, n);
  const s = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s[i * n + j] = (m[i * n + j] + m[j * n + i]) * 0.5;
  return s;
}

function spd(n, seed) {
  const m = det2(seed, n, n);
  const a = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += m[k * n + i] * m[k * n + j];
    a[i * n + j] = s + (i === j ? n : 0);
  }
  return a;
}

const pair = (data, shape) => [tensor(data, { shape, dtype: 'f32' }), tensor(data, { shape, dtype: 'f32' }).to(GPU_DEVICE)];
const diag = (vals, device) => {
  const k = vals.length;
  const d = new Float32Array(k * k);
  for (let i = 0; i < k; i++) d[i * k + i] = vals[i];
  return tensor(d, { shape: [k, k], dtype: 'f32', device });
};

describe.skipIf(!cudaDeps)('CUDA cuSOLVER linalg matches CPU reference (f32)', () => {
  beforeAll(async () => { await preloadCudaRuntime(); }, 60000);

  describe('svd (reduced) — reconstruction and singular values', () => {
    for (const [name, m, n] of [['square 6x6', 6, 6], ['tall 8x5', 8, 5], ['wide 5x8', 5, 8]]) {
      it(`${name}`, () => {
        const data = det2(1, m, n);
        const [c, g] = pair(data, [m, n]);
        const sc = linalg.svd(c), sg = linalg.svd(g);
        expect(sg.U.shape).toEqual([m, Math.min(m, n)]);
        expect(sg.V.shape).toEqual([n, Math.min(m, n)]);
        expect(maxAbsErr(flat(sc.S), flat(sg.S)), 'singular values').toBeLessThan(TOL);
        const recon = matmul(matmul(sg.U, diag(flat(sg.S), sg.U.device)), sg.V.transpose(0, 1));
        expect(maxAbsErr(flat(recon), Array.from(data)), 'reconstruction A=USVᵀ').toBeLessThan(TOL);
      }, 60000);
    }
  });

  describe('eigh (symmetric) — eigenvalues and reconstruction', () => {
    for (const n of [4, 7]) {
      it(`${n}x${n} symmetric`, () => {
        const data = symmetric(n, 2);
        const [c, g] = pair(data, [n, n]);
        const ec = linalg.eigh(c), eg = linalg.eigh(g);
        expect(maxAbsErr(flat(ec.values), flat(eg.values)), 'eigenvalues (ascending)').toBeLessThan(TOL);
        const recon = matmul(matmul(eg.vectors, diag(flat(eg.values), eg.vectors.device)), eg.vectors.transpose(0, 1));
        expect(maxAbsErr(flat(recon), Array.from(data)), 'reconstruction VΛVᵀ').toBeLessThan(TOL);
      }, 60000);
    }
  });

  it('cholesky (SPD) matches CPU, A = L Lᵀ', () => {
    const n = 6;
    const data = spd(n, 3);
    const [c, g] = pair(data, [n, n]);
    const lc = flat(linalg.cholesky(c)), lg = flat(linalg.cholesky(g));
    expect(maxAbsErr(lc, lg), 'L factor').toBeLessThan(TOL);
    const recon = matmul(linalg.cholesky(g), linalg.cholesky(g).transpose(0, 1));
    expect(maxAbsErr(flat(recon), Array.from(data)), 'L Lᵀ').toBeLessThan(TOL);
  }, 60000);

  it('solve — vector rhs', () => {
    const n = 6;
    const A = spd(n, 4);
    const b = det2(5, n, 1);
    const [Ac, Ag] = pair(A, [n, n]);
    const bc = tensor(b, { shape: [n], dtype: 'f32' });
    const bg = tensor(b, { shape: [n], dtype: 'f32' }).to(GPU_DEVICE);
    const xg = linalg.solve(Ag, bg);
    expect(xg.shape).toEqual([n]);
    expect(maxAbsErr(flat(linalg.solve(Ac, bc)), flat(xg))).toBeLessThan(TOL);
  }, 60000);

  it('solve — matrix rhs', () => {
    const n = 5, nrhs = 3;
    const A = spd(n, 6);
    const B = det2(7, n, nrhs);
    const [Ac, Ag] = pair(A, [n, n]);
    const Bc = tensor(B, { shape: [n, nrhs], dtype: 'f32' });
    const Bg = tensor(B, { shape: [n, nrhs], dtype: 'f32' }).to(GPU_DEVICE);
    const xg = linalg.solve(Ag, Bg);
    expect(xg.shape).toEqual([n, nrhs]);
    expect(maxAbsErr(flat(linalg.solve(Ac, Bc)), flat(xg))).toBeLessThan(TOL);
  }, 60000);

  it('inv matches CPU and A·A⁻¹ = I', () => {
    const n = 6;
    const A = spd(n, 8);
    const [Ac, Ag] = pair(A, [n, n]);
    expect(maxAbsErr(flat(linalg.inv(Ac)), flat(linalg.inv(Ag)))).toBeLessThan(TOL);
    const id = flat(matmul(Ag, linalg.inv(Ag)));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) expect(Math.abs(id[i * n + j] - (i === j ? 1 : 0))).toBeLessThan(TOL);
  }, 60000);

  it('det matches CPU', () => {
    const n = 5;
    const A = spd(n, 9);
    const [Ac, Ag] = pair(A, [n, n]);
    expect(Math.abs(linalg.det(Ac) - linalg.det(Ag)) / Math.abs(linalg.det(Ac))).toBeLessThan(TOL);
  }, 60000);

  describe('pinv (full-rank) matches CPU', () => {
    for (const [name, m, n] of [['tall 7x4', 7, 4], ['wide 4x7', 4, 7]]) {
      it(name, () => {
        const data = det2(10, m, n);
        const [c, g] = pair(data, [m, n]);
        const pc = flat(linalg.pinv(c)), pg = flat(linalg.pinv(g));
        expect(pg.length).toBe(n * m);
        expect(relErr(pc, pg)).toBeLessThan(TOL);
      }, 60000);
    }
  });

  it('lstsq (overdetermined full-rank) matches CPU', () => {
    const m = 8, n = 3;
    const A = det2(11, m, n);
    const b = det2(12, m, 1);
    const [Ac, Ag] = pair(A, [m, n]);
    const bc = tensor(b, { shape: [m], dtype: 'f32' });
    const bg = tensor(b, { shape: [m], dtype: 'f32' }).to(GPU_DEVICE);
    const xg = linalg.lstsq(Ag, bg);
    expect(xg.shape).toEqual([n]);
    expect(maxAbsErr(flat(linalg.lstsq(Ac, bc)), flat(xg))).toBeLessThan(TOL);
  }, 60000);

  it('cov matches CPU', () => {
    const rows = 20, cols = 5;
    const data = det2(13, rows, cols);
    const [c, g] = pair(data, [rows, cols]);
    expect(maxAbsErr(flat(linalg.cov(c)), flat(linalg.cov(g)))).toBeLessThan(TOL);
  }, 60000);
});

describe.skipIf(!cudaDeps)('CUDA ml estimators (linalg-backed) match CPU on GPU tensors', () => {
  beforeAll(async () => { await preloadCudaRuntime(); }, 60000);

  const N = 48, D = 4;
  const Xflat = det2(20, N, D);
  const yd = new Float32Array(N);
  for (let i = 0; i < N; i++) { let t = 0; for (let j = 0; j < D; j++) t += (j + 1) * Xflat[i * D + j]; yd[i] = t + 0.03 * Math.cos(i); }
  const mkX = (dev) => tensor(Xflat, { shape: [N, D], dtype: 'f32', ...(dev ? { device: dev } : {}) });
  const mkY = (dev) => tensor(yd, { shape: [N], dtype: 'f32', ...(dev ? { device: dev } : {}) });

  it('Ridge fit + predict', () => {
    const rc = new ml.Ridge({ alpha: 1 }).fit(mkX(), mkY());
    const rg = new ml.Ridge({ alpha: 1 }).fit(mkX(GPU_DEVICE), mkY(GPU_DEVICE));
    expect(relErr(flat(rc.coef_), flat(rg.coef_))).toBeLessThan(TOL);
    expect(relErr(flat(rc.predict(mkX())), flat(rg.predict(mkX(GPU_DEVICE))))).toBeLessThan(TOL);
  }, 60000);

  it('LinearRegression (lstsq) fit + predict', () => {
    const rc = new ml.LinearRegression().fit(mkX(), mkY());
    const rg = new ml.LinearRegression().fit(mkX(GPU_DEVICE), mkY(GPU_DEVICE));
    expect(relErr(flat(rc.predict(mkX())), flat(rg.predict(mkX(GPU_DEVICE))))).toBeLessThan(TOL);
  }, 60000);

  it('PCA fit + transform + explained variance', () => {
    const pc = new ml.PCA({ nComponents: 2 }).fit(mkX());
    const pg = new ml.PCA({ nComponents: 2 }).fit(mkX(GPU_DEVICE));
    const ac = flat(pc.transform(mkX())).map(Math.abs);
    const ag = flat(pg.transform(mkX(GPU_DEVICE))).map(Math.abs);
    expect(relErr(ac, ag)).toBeLessThan(TOL);
    for (let c = 0; c < 2; c++) expect(Math.abs(pc.explainedVarianceRatio_[c] - pg.explainedVarianceRatio_[c])).toBeLessThan(TOL);
  }, 60000);

  it('LogisticRegression fit + predict', () => {
    const lbl = new Float32Array(N);
    for (let i = 0; i < N; i++) lbl[i] = yd[i] > 0 ? 1 : 0;
    const yc = tensor(lbl, { shape: [N], dtype: 'f32' });
    const lc = new ml.LogisticRegression({ maxIter: 300 }).fit(mkX(), yc);
    const lg = new ml.LogisticRegression({ maxIter: 300 }).fit(mkX(GPU_DEVICE), yc);
    const pc = flat(lc.predict(mkX())), pg = flat(lg.predict(mkX(GPU_DEVICE)));
    let agree = 0;
    for (let i = 0; i < N; i++) if (pc[i] === pg[i]) agree++;
    expect(agree).toBe(N);
    expect(relErr(flat(lc.W_), flat(lg.W_))).toBeLessThan(TOL);
  }, 60000);
});

describe.skipIf(!cudaDeps)('CUDA linalg benchmark vs CPU (large matrices)', () => {
  beforeAll(async () => { await preloadCudaRuntime(); }, 60000);

  const timeit = (fn, reps = 3) => {
    let best = Infinity;
    for (let r = 0; r < reps; r++) { const t0 = performance.now(); fn(); best = Math.min(best, performance.now() - t0); }
    return best;
  };

  for (const n of [512, 1024]) {
    it(`solve ${n}x${n}: GPU vs CPU`, () => {
      const A = spd(n, 42);
      const b = det2(43, n, 1);
      const Ac = tensor(A, { shape: [n, n], dtype: 'f32' });
      const bc = tensor(b, { shape: [n], dtype: 'f32' });
      const Ag = tensor(A, { shape: [n, n], dtype: 'f32' }).to(GPU_DEVICE);
      const bg = tensor(b, { shape: [n], dtype: 'f32' }).to(GPU_DEVICE);
      linalg.solve(Ag, bg);
      const cpuMs = timeit(() => linalg.solve(Ac, bc));
      const gpuMs = timeit(() => linalg.solve(Ag, bg));
      const xc = flat(linalg.solve(Ac, bc)), xg = flat(linalg.solve(Ag, bg));
      expect(maxAbsErr(xc, xg) / (Math.max(...xc.map(Math.abs)) || 1)).toBeLessThan(1e-2);
      console.log(`  solve ${n}x${n}: CPU ${cpuMs.toFixed(1)}ms  GPU ${gpuMs.toFixed(1)}ms  speedup ${(cpuMs / gpuMs).toFixed(2)}x`);
    }, 120000);
  }

  it('cholesky 1024x1024: GPU vs CPU', () => {
    const n = 1024;
    const A = spd(n, 44);
    const Ac = tensor(A, { shape: [n, n], dtype: 'f32' });
    const Ag = tensor(A, { shape: [n, n], dtype: 'f32' }).to(GPU_DEVICE);
    linalg.cholesky(Ag);
    const cpuMs = timeit(() => linalg.cholesky(Ac), 2);
    const gpuMs = timeit(() => linalg.cholesky(Ag), 2);
    console.log(`  cholesky ${n}x${n}: CPU ${cpuMs.toFixed(1)}ms  GPU ${gpuMs.toFixed(1)}ms  speedup ${(cpuMs / gpuMs).toFixed(2)}x`);
    expect(maxAbsErr(flat(linalg.cholesky(Ac)), flat(linalg.cholesky(Ag)))).toBeLessThan(1e-1);
  }, 120000);
});
