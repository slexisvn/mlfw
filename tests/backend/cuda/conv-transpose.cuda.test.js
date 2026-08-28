import { describe, it, expect } from 'vitest';
import { tensor, ConvTranspose2d, ConvTranspose1d, compile } from '../../../src/index.js';
import { CUDATarget } from '../../../src/backend/target.js';
import { cudaDeps } from '../../_utils/cuda.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat } from '../../_utils/tensor_data.js';

function maxRelErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / (1 + Math.abs(b[i])));
  return m;
}

describe.skipIf(!cudaDeps)('ConvTranspose on real GPU', () => {
  const CASES = [
    { name: '2d stride 2 padded', build: () => new ConvTranspose2d(2, 3, 3, { stride: 2, padding: 1 }), shape: [1, 2, 4, 4], tol: 0 },
    { name: '2d stride 3', build: () => new ConvTranspose2d(2, 3, 3, { stride: 3, padding: 1 }), shape: [1, 2, 4, 4], tol: 1e-6 },
    { name: '2d output padding', build: () => new ConvTranspose2d(2, 4, 3, { stride: 2, padding: 1, outputPadding: 1 }), shape: [1, 2, 5, 5], tol: 1e-6 },
    { name: '1d stride 2', build: () => new ConvTranspose1d(2, 3, 3, { stride: 2, padding: 1 }), shape: [1, 2, 6], tol: 1e-6 },
  ];

  for (const c of CASES) {
    it(`${c.name} matches eager`, async () => {
      const model = c.build();
      const x = tensor(randomNested(mulberry32(c.name.length * 31 + c.shape[2]), c.shape));

      const fn = compile(model, [x], { target: CUDATarget(), scheduling: { enabled: false } });
      const gpu = flat(await fn(x));
      const eager = flat(model.forward(x));

      expect(gpu.length).toBe(eager.length);
      expect(maxRelErr(gpu, eager)).toBeLessThanOrEqual(c.tol);
    });
  }

  it('emits a kernel small enough for NVRTC to compile promptly', async () => {
    const model = new ConvTranspose2d(2, 3, 3, { stride: 2, padding: 1 });
    const x = tensor(randomNested(mulberry32(7), [1, 2, 4, 4]));
    const fn = compile(model, [x], { target: CUDATarget(), scheduling: { enabled: false } });

    const started = performance.now();
    await fn(x);
    const elapsed = performance.now() - started;

    expect(fn.source().length).toBeLessThan(1e6);
    expect(elapsed).toBeLessThan(15000);
  });
});
