import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../../_utils/tensor_data.js';

const EPS = 1e-3;
const TOL = 5e-3;

const CASES = [
  { name: 'causal, square attention', shapes: [[1, 2, 5, 4], [1, 2, 5, 4], [1, 2, 5, 4]], causal: true },
  { name: 'causal, more keys than queries', shapes: [[1, 2, 3, 4], [1, 2, 6, 4], [1, 2, 6, 4]], causal: true },
  { name: 'non-causal cross attention', shapes: [[1, 2, 4, 4], [1, 2, 5, 4], [1, 2, 5, 4]], causal: false },
];

describe('scaled_dot_product_attention VJP matches finite differences', () => {
  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 31 + 7);
      const datas = c.shapes.map((s) => randomNested(rng, s));
      const fwd = (q, k, v) => nn.F.scaled_dot_product_attention(q, k, v, null, 0, c.causal);

      const inputs = datas.map((d) => tensor(d));
      const cf = compileWithBackward({ forward: (...a) => fwd(...a) }, inputs, { target: CPUTarget() });
      const out = cf(...inputs);
      const analytic = cf.backward(ones(out.shape)).map(flat);

      expect(analytic.length).toBe(3);
      for (let argi = 0; argi < 3; argi++) {
        const base = flat(tensor(datas[argi]));
        const n = numel(c.shapes[argi]);
        const step = Math.max(1, Math.floor(n / 10));
        for (let k = 0; k < n; k += step) {
          const sumAt = (delta) => {
            const arr = Array.from(base);
            arr[k] += delta;
            const args = datas.map((d, i) => tensor(i === argi ? nest(arr, c.shapes[argi]) : d));
            return flat(fwd(...args)).reduce((a, b) => a + b, 0);
          };
          const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
          const relErr = Math.abs(numeric - analytic[argi][k]) / (1 + Math.abs(numeric));
          expect(relErr, `${c.name} arg${argi}[${k}]: numeric=${numeric} analytic=${analytic[argi][k]}`).toBeLessThan(TOL);
        }
      }
    }, 60000);
  }

  it('causal masking actually changes the gradient', () => {
    const rng = mulberry32(4242);
    const shapes = [[1, 1, 5, 4], [1, 1, 5, 4], [1, 1, 5, 4]];
    const datas = shapes.map((s) => randomNested(rng, s));
    const gradsFor = (causal) => {
      const inputs = datas.map((d) => tensor(d));
      const cf = compileWithBackward(
        { forward: (q, k, v) => nn.F.scaled_dot_product_attention(q, k, v, null, 0, causal) },
        inputs, { target: CPUTarget() },
      );
      const out = cf(...inputs);
      return cf.backward(ones(out.shape)).map(flat);
    };

    const causal = gradsFor(true);
    const full = gradsFor(false);
    const differs = causal[0].some((v, i) => Math.abs(v - full[0][i]) > 1e-4);
    expect(differs, 'causal and non-causal gradients should differ').toBe(true);
  }, 60000);
});
