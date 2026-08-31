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
  { name: 'softmax', shape: [2, 4, 5], fwd: (x) => nn.F.softmax(x, -1) },
  { name: 'log_softmax', shape: [2, 4, 5], fwd: (x) => nn.F.log_softmax(x, -1) },
  { name: 'softmax then scale', shape: [3, 6], fwd: (x) => nn.F.softmax(x, -1).mul(2.5) },
  { name: 'mean-centred', shape: [2, 3, 4], fwd: (x) => x.sub(x.mean(-1, true)) },
  { name: 'sum-normalised', shape: [2, 3, 4], fwd: (x) => x.div(x.sum(-1, true).add(3)) },
  { name: 'reduce over leading axis', shape: [3, 4], fwd: (x) => x.sub(x.sum(0, true)) },
];

describe('reduction fusion VJP matches central finite differences', () => {
  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 41 + 3);
      const data = randomNested(rng, c.shape);
      const input = tensor(data);
      const cf = compileWithBackward({ forward: c.fwd }, [input], { target: CPUTarget() });
      const out = cf(input);
      const [analytic] = cf.backward(ones(out.shape)).map(flat);

      const base = flat(tensor(data));
      const n = numel(c.shape);
      const step = Math.max(1, Math.floor(n / 8));
      for (let k = 0; k < n; k += step) {
        const sumAt = (delta) => {
          const arr = Array.from(base);
          arr[k] += delta;
          return flat(c.fwd(tensor(nest(arr, c.shape)))).reduce((a, b) => a + b, 0);
        };
        const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
        const relErr = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
        expect(relErr, `${c.name}[${k}]: numeric=${numeric} analytic=${analytic[k]}`).toBeLessThan(TOL);
      }
    }, 60000);
  }

  it('attention head gradients match finite differences', () => {
    class Head extends nn.Module {
      constructor(dim) {
        super();
        this.q = new nn.Linear(dim, dim);
        this.k = new nn.Linear(dim, dim);
        this.v = new nn.Linear(dim, dim);
        this.scale = 1 / Math.sqrt(dim);
      }
      forward(x) {
        const q = this.q.forward(x);
        const k = this.k.forward(x);
        const v = this.v.forward(x);
        const scores = q.matmul(k.transpose(-2, -1)).mul(this.scale);
        return nn.F.softmax(scores, -1).matmul(v);
      }
    }

    const shape = [1, 4, 4];
    const rng = mulberry32(9091);
    const model = new Head(4);
    const data = randomNested(rng, shape);
    const input = tensor(data);
    const cf = compileWithBackward(model, [input], { target: CPUTarget() });
    const out = cf(input);
    const [analytic] = cf.backward(ones(out.shape)).map(flat);

    const base = flat(tensor(data));
    const n = numel(shape);
    for (let k = 0; k < n; k += 3) {
      const sumAt = (delta) => {
        const arr = Array.from(base);
        arr[k] += delta;
        return flat(model.forward(tensor(nest(arr, shape)))).reduce((a, b) => a + b, 0);
      };
      const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
      const relErr = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
      expect(relErr, `head[${k}]: numeric=${numeric} analytic=${analytic[k]}`).toBeLessThan(TOL);
    }
  }, 60000);
});
