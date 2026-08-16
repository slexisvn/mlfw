import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../../_utils/tensor_data.js';

const EPS = 2e-3;
const TOL = 5e-3;

const CASES = [
  { name: 'LSTM single layer', make: () => { const m = new nn.LSTM(3, 4, 1, true); return (x) => m.forward(x)[0]; }, shape: [1, 4, 3] },
  { name: 'GRU single layer', make: () => { const m = new nn.GRU(3, 4, 1, true); return (x) => m.forward(x)[0]; }, shape: [1, 4, 3] },
  { name: 'LSTM two layers', make: () => { const m = new nn.LSTM(3, 4, 2, true); return (x) => m.forward(x)[0]; }, shape: [1, 3, 3] },
];

describe.each(['separate', 'joint'])('scan VJP (%s mode) matches finite differences', (mode) => {
  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 13 + 1);
      const data = randomNested(rng, c.shape);
      const fwd = c.make();

      const inputs = [tensor(data)];
      const cf = compileWithBackward({ forward: (x) => fwd(x) }, inputs, { target: CPUTarget(), mode });
      const out = cf(...inputs);
      const analytic = flat(cf.backward(ones(out.shape))[0]);

      const base = flat(tensor(data));
      const n = numel(c.shape);
      expect(analytic.length).toBe(n);

      for (let k = 0; k < n; k++) {
        const sumAt = (delta) => {
          const arr = Array.from(base);
          arr[k] += delta;
          return flat(fwd(tensor(nest(arr, c.shape)))).reduce((a, b) => a + b, 0);
        };
        const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
        const relErr = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
        expect(relErr, `${c.name}[${k}]: numeric=${numeric} analytic=${analytic[k]}`).toBeLessThan(TOL);
      }
    }, 120000);
  }
});
