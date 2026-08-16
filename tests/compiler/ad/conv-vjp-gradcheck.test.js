import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../../src/backend/target.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../../_utils/tensor_data.js';

const CASES = [
  { name: 'stride 1, dilation 1', xs: [1, 2, 5, 5], w: [3, 2, 3, 3], stride: 1, padding: 1 },
  { name: 'stride 2', xs: [1, 2, 6, 6], w: [3, 2, 3, 3], stride: 2, padding: 1 },
  { name: 'stride 2 with ragged trailing input', xs: [1, 2, 7, 7], w: [3, 2, 3, 3], stride: 2, padding: 1 },
  { name: 'dilation 2', xs: [1, 2, 7, 7], w: [3, 2, 3, 3], stride: 1, padding: 2, dilation: 2 },
  { name: 'stride 2 + dilation 2', xs: [1, 2, 9, 9], w: [2, 2, 3, 3], stride: 2, padding: 2, dilation: 2 },
  { name: 'groups 2', xs: [1, 4, 5, 5], w: [4, 2, 3, 3], stride: 1, padding: 1, groups: 2 },
  { name: 'groups 2 + stride 2', xs: [1, 4, 6, 6], w: [6, 2, 3, 3], stride: 2, padding: 1, groups: 2 },
  { name: 'conv1d dilation 2', xs: [1, 2, 10], w: [3, 2, 3], padding: 2, dilation: 2, oneD: true },
  { name: 'conv1d stride 2', xs: [1, 2, 10], w: [3, 2, 3], stride: 2, padding: 1, oneD: true },
  { name: 'conv1d groups 2', xs: [1, 4, 10], w: [4, 2, 3], padding: 1, groups: 2, oneD: true },
];

const EPS = 1e-3;
const TOL = 5e-3;

function forwardOf(c) {
  const fn = c.oneD ? nn.F.conv1d : nn.F.conv2d;
  return (x, w) => fn(x, w, null, c.stride ?? 1, c.padding ?? 0, c.dilation ?? 1, c.groups ?? 1);
}

describe('conv VJP matches finite differences', () => {
  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 977 + c.xs[2]);
      const shapes = [c.xs, c.w];
      const datas = shapes.map((s) => randomNested(rng, s));
      const fwd = forwardOf(c);

      const inputs = datas.map((d) => tensor(d));
      const cf = compileWithBackward({ forward: (a, b) => fwd(a, b) }, inputs, { target: CPUTarget() });
      const out = cf(...inputs);
      const analytic = cf.backward(ones(out.shape)).map(flat);

      expect(analytic.length).toBe(2);
      for (let argi = 0; argi < 2; argi++) {
        expect(analytic[argi].length).toBe(numel(shapes[argi]));

        const base = flat(tensor(datas[argi]));
        const n = numel(shapes[argi]);
        const step = Math.max(1, Math.floor(n / 12));
        for (let k = 0; k < n; k += step) {
          const sumAt = (delta) => {
            const arr = Array.from(base);
            arr[k] += delta;
            const args = datas.map((d, i) => tensor(i === argi ? nest(arr, shapes[argi]) : d));
            return flat(fwd(...args)).reduce((a, b) => a + b, 0);
          };
          const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
          const relErr = Math.abs(numeric - analytic[argi][k]) / (1 + Math.abs(numeric));
          expect(relErr, `${c.name} arg${argi}[${k}]: numeric=${numeric} analytic=${analytic[argi][k]}`).toBeLessThan(TOL);
        }
      }
    }, 60000);
  }
});
