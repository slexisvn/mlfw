import { describe, it, expect } from 'vitest';
import { tensor, sum } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import * as ops from '../../src/tensor/ops/ops.js';
import { dispatcher } from '../../src/dispatcher/dispatcher.js';
import { listOpsWithoutGrad } from '../../src/autograd/registry.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../_utils/tensor_data.js';

const EPS = 1e-3;
const TOL = 3e-3;

function gradcheck(fwd, shapes, seed, lo = -1, hi = 1) {
  const rng = mulberry32(seed);
  const datas = shapes.map((s) => randomNested(rng, s, lo, hi));
  const inputs = datas.map((d) => tensor(d, { requiresGrad: true }));
  const out = fwd(...inputs);
  expect(out.requiresGrad, 'forward output must track gradients').toBe(true);
  sum(out).backward();

  for (let argi = 0; argi < shapes.length; argi++) {
    const analytic = flat(inputs[argi].grad);
    expect(analytic.length, `arg${argi} gradient size`).toBe(numel(shapes[argi]));

    const base = flat(tensor(datas[argi]));
    const n = numel(shapes[argi]);
    const step = Math.max(1, Math.floor(n / 8));
    for (let k = 0; k < n; k += step) {
      const sumAt = (delta) => {
        const arr = Array.from(base);
        arr[k] += delta;
        const args = datas.map((d, i) => tensor(i === argi ? nest(arr, shapes[argi]) : d));
        return flat(fwd(...args)).reduce((a, b) => a + b, 0);
      };
      const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
      const relErr = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
      expect(relErr, `arg${argi}[${k}]: numeric=${numeric} analytic=${analytic[k]}`).toBeLessThan(TOL);
    }
  }
}

describe('every dispatched op has a gradient path', () => {
  it('no op silently drops the gradient', () => {
    const names = [...new Set(dispatcher.listOps().map((k) => dispatcher.findOp(k).name))];
    expect(listOpsWithoutGrad(names)).toEqual([]);
  });

  it('an op with no rule and no barrier fails loudly instead of dropping the gradient', async () => {
    const { getGradFn } = await import('../../src/autograd/registry.js');
    expect(() => getGradFn('definitely_not_a_registered_op')).toThrow(/silently dropped/);
  });
});

describe('eager gradients match finite differences', () => {
  it('abs', () => gradcheck((x) => ops.abs(x), [[3, 4]], 1));
  it('sin and cos', () => gradcheck((x) => ops.mul(ops.sin(x), ops.cos(x)), [[3, 4]], 2));
  it('rsqrt', () => gradcheck((x) => ops.rsqrt(ops.add(ops.mul(x, x), 1)), [[3, 4]], 3));
  it('max reduction', () => gradcheck((x) => ops.max(x, -1), [[3, 4]], 4));
  it('min reduction', () => gradcheck((x) => ops.min(x, -1), [[3, 4]], 5));
  it('prod reduction', () => gradcheck((x) => ops.prod(ops.add(x, 2), -1), [[3, 4]], 6));
  it('maximum', () => gradcheck((a, b) => ops.maximum(a, b), [[3, 4], [3, 4]], 8));
  it('minimum', () => gradcheck((a, b) => ops.minimum(a, b), [[3, 4], [3, 4]], 9));
  it('clone', () => gradcheck((x) => ops.clone(x), [[3, 4]], 13));
  it('contiguous after transpose', () => gradcheck((x) => ops.contiguous(ops.transpose(x, 0, 1)), [[3, 4]], 14));

  it('cumsum through its decomposition', () => gradcheck((x) => ops.cumsum(x, -1), [[3, 4]], 7));
  it('flip through its decomposition', () => gradcheck((x) => ops.mul(ops.flip(x, [-1]), x), [[3, 4]], 10));
  it('roll through its decomposition', () => gradcheck((x) => ops.mul(ops.roll(x, 1, -1), x), [[3, 4]], 11));
  it('repeat through its decomposition', () => gradcheck((x) => ops.repeat(x, [2, 3]), [[2, 3]], 12));
  it('split through its decomposition', () => gradcheck((x) => {
    const parts = ops.split(x, [2, 2], -1);
    return ops.mul(parts[0], parts[1]);
  }, [[3, 4]], 15));
});

describe('neural-network layers produce eager gradients', () => {
  it('conv2d', () => gradcheck((x, w) => nn.F.conv2d(x, w, null, 1, 1, 1, 1), [[1, 2, 5, 5], [3, 2, 3, 3]], 20));
  it('conv2d with stride 2', () => gradcheck((x, w) => nn.F.conv2d(x, w, null, 2, 1, 1, 1), [[1, 2, 6, 6], [3, 2, 3, 3]], 21));
  it('conv2d with dilation 2', () => gradcheck((x, w) => nn.F.conv2d(x, w, null, 1, 2, 2, 1), [[1, 2, 7, 7], [3, 2, 3, 3]], 26));
  it('conv2d with 2 groups', () => gradcheck((x, w) => nn.F.conv2d(x, w, null, 1, 1, 1, 2), [[1, 4, 5, 5], [4, 2, 3, 3]], 22));
  it('conv1d', () => gradcheck((x, w) => nn.F.conv1d(x, w, null, 1, 1, 1, 1), [[1, 2, 8], [3, 2, 3]], 27));
  it('max pooling', () => gradcheck((x) => nn.F.max_pool2d(x, 2, 2), [[1, 2, 4, 4]], 23));
  it('average pooling', () => gradcheck((x) => nn.F.avg_pool2d(x, 2, 2), [[1, 2, 4, 4]], 24));
  it('layer norm', () => gradcheck((x, w, b) => ops.layer_norm(x, w, b, -1, 1e-5), [[3, 4], [4], [4]], 25));

  it('embedding accumulates into the table', () => {
    const table = tensor([[1, 2], [3, 4], [5, 6]], { requiresGrad: true });
    const indices = tensor([0, 2, 0]);
    sum(ops.embedding(table, indices)).backward();
    expect(flat(table.grad)).toEqual([2, 2, 0, 0, 1, 1]);
  });

  it('a CNN classifier trains end to end', () => {
    const conv = new nn.Conv2d(1, 2, 3, { padding: 1 });
    const pool = new nn.MaxPool2d(2), flatten = new nn.Flatten(), fc = new nn.Linear(2 * 2 * 2, 2);
    const rng = mulberry32(99);
    const x = tensor(randomNested(rng, [2, 1, 4, 4]));
    const target = tensor(randomNested(rng, [2, 2]));
    const loss = new nn.MSELoss();

    const step = () => loss.forward(fc.forward(flatten.forward(pool.forward(nn.F.relu(conv.forward(x))))), target);
    const before = flat(step())[0];
    for (let i = 0; i < 12; i++) {
      const l = step();
      l.backward();
      for (const p of [conv.weight, conv.bias, fc.weight, fc.bias]) {
        if (!p || !p.grad) continue;
        const updated = ops.sub(p.detach(), ops.mul(p.grad, 0.05));
        p._impl.storage.data.set(flat(updated));
        p.grad = null;
      }
    }
    const after = flat(step())[0];
    expect(after, `loss ${before} -> ${after}`).toBeLessThan(before);
  });
});
