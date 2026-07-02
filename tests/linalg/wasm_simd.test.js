import { describe, it, expect } from 'vitest';
import { tensor, linalg } from '../../src/index.js';
import { WASM_DEVICE, CPU_DEVICE } from '../../src/tensor/types/device.js';
import { cpuSvd } from '../../src/kernels/cpu/linalg/ops.js';
import { wasmSvd } from '../../src/kernels/wasm/linalg/svd.js';
import { makeRng } from '../../src/ml/_random.js';

function randMat(rows, cols, seed, device) {
  const rng = makeRng(seed);
  const a = new Float64Array(rows * cols);
  for (let i = 0; i < a.length; i++) a[i] = rng() * 2 - 1;
  return tensor(a, { shape: [rows, cols], device });
}

function reconError(U, S, V, A) {
  const rows = U.length, cols = V.length, k = S.length;
  let m = 0;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let c = 0; c < k; c++) s += U[i][c] * S[c] * V[j][c];
      m = Math.max(m, Math.abs(s - A[i][j]));
    }
  }
  return m;
}

describe('wasm-simd linalg svd matches the js reference on wasm-device tensors', () => {
  for (const [rows, cols] of [[400, 40], [80, 120]]) {
    it(`svd (${rows}x${cols}, above threshold) matches the js reference`, () => {
      const A = randMat(rows, cols, 7, WASM_DEVICE);
      const js = cpuSvd([], A);
      const wa = wasmSvd([], A);
      const sjs = js[1].toArray();
      const swa = wa[1].toArray();
      let sd = 0;
      for (let i = 0; i < sjs.length; i++) sd = Math.max(sd, Math.abs(sjs[i] - swa[i]));
      expect(sd).toBeLessThan(1e-5);
      expect(reconError(wa[0].toArray(), swa, wa[2].toArray(), A.toArray())).toBeLessThan(1e-4);
      expect(wa[1].device.type).toBe('wasm');
    });
  }

  it('dispatches through public linalg.svd on a wasm-device tensor', () => {
    const A = randMat(400, 30, 13, WASM_DEVICE);
    const r = linalg.svd(A);
    const Aarr = A.toArray();
    expect(reconError(r.U.toArray(), r.S.toArray(), r.V.toArray(), Aarr)).toBeLessThan(1e-4);
    expect(r.S.device.type).toBe('wasm');
  });

  it('small matrix falls back to the js kernel (equivalent results)', () => {
    const data = [1, 1.1, 2, 1.9, 3, 3.2, 4, 3.8, 5, 5.1];
    const Aw = tensor(Float64Array.from(data), { shape: [5, 2], device: WASM_DEVICE });
    const Ac = tensor(Float64Array.from(data), { shape: [5, 2], device: CPU_DEVICE });
    const sw = wasmSvd([], Aw)[1].toArray();
    const sc = cpuSvd([], Ac)[1].toArray();
    for (let i = 0; i < sw.length; i++) expect(sw[i]).toBeCloseTo(sc[i], 6);
  });
});
