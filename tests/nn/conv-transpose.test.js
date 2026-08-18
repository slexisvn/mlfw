import { describe, it, expect } from 'vitest';
import { tensor, compile, compileWithBackward, CPUTarget, WasmTarget, CUDATarget } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../_utils/tensor_data.js';

const F = nn.F;

function maxRelErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / (1 + Math.abs(a[i])));
  return m;
}

describe('conv_transpose2d equals the input-gradient of conv2d (independent oracle)', () => {
  const CASES = [
    { name: 'stride 1', x: [1, 2, 5, 5], w: [3, 2, 3, 3], stride: 1, padding: 0 },
    { name: 'stride 1 padded', x: [1, 2, 5, 5], w: [3, 2, 3, 3], stride: 1, padding: 1 },
    { name: 'stride 2', x: [1, 2, 6, 6], w: [3, 2, 3, 3], stride: 2, padding: 1 },
    { name: 'stride 2 asymmetric kernel', x: [1, 2, 6, 8], w: [4, 2, 3, 3], stride: 2, padding: 1 },
    { name: 'dilation 2', x: [1, 2, 7, 7], w: [3, 2, 3, 3], stride: 1, padding: 2, dilation: 2 },
    { name: 'groups 2', x: [2, 4, 5, 5], w: [4, 2, 3, 3], stride: 1, padding: 1, groups: 2 },
  ];

  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 131 + c.x[3]);
      const xd = randomNested(rng, c.x), wd = randomNested(rng, c.w);
      const x = tensor(xd), w = tensor(wd);
      const stride = c.stride ?? 1, padding = c.padding ?? 0, dilation = c.dilation ?? 1, groups = c.groups ?? 1;

      const fwd = (xx, ww) => F.conv2d(xx, ww, null, stride, padding, dilation, groups);
      const cf = compileWithBackward({ forward: fwd }, [x, w], { target: CPUTarget() });
      const y = cf(x, w);
      const dx = flat(cf.backward(ones(y.shape))[0]);

      const g = ones(y.shape);
      const recoverPad = (d) => (c.x[d] + 2 * padding - dilation * (c.w[d] - 1) - 1) % stride;
      const outputPadding = [recoverPad(2), recoverPad(3)];
      const viaTranspose = flat(F.conv_transpose2d(g, w, null, stride, padding, outputPadding, dilation, groups));

      expect(viaTranspose.length).toBe(dx.length);
      expect(maxRelErr(dx, viaTranspose)).toBeLessThan(1e-5);
    });
  }
});

describe('ConvTranspose output shape follows (in-1)*stride - 2*padding + dilation*(k-1) + output_padding + 1', () => {
  const CASES = [
    { in: 4, stride: 1, padding: 0, k: 3, dilation: 1, outputPadding: 0 },
    { in: 4, stride: 2, padding: 0, k: 3, dilation: 1, outputPadding: 0 },
    { in: 4, stride: 2, padding: 1, k: 3, dilation: 1, outputPadding: 1 },
    { in: 5, stride: 3, padding: 2, k: 4, dilation: 1, outputPadding: 2 },
    { in: 4, stride: 2, padding: 2, k: 3, dilation: 2, outputPadding: 0 },
  ];

  for (const c of CASES) {
    it(`in=${c.in} s=${c.stride} p=${c.padding} k=${c.k} d=${c.dilation} op=${c.outputPadding}`, () => {
      const m = new nn.ConvTranspose2d(2, 3, c.k, {
        stride: c.stride, padding: c.padding, outputPadding: c.outputPadding, dilation: c.dilation,
      });
      const rng = mulberry32(c.in * 17 + c.stride);
      const out = m.forward(tensor(randomNested(rng, [1, 2, c.in, c.in])));
      const expected = (c.in - 1) * c.stride - 2 * c.padding + c.dilation * (c.k - 1) + c.outputPadding + 1;
      expect(out.shape).toEqual([1, 3, expected, expected]);
      expect(flat(out).every(Number.isFinite)).toBe(true);
    });
  }
});

describe('ConvTranspose1d equals the 2d form on a singleton height', () => {
  it('matches conv_transpose2d with an unsqueezed spatial axis', () => {
    const rng = mulberry32(4242);
    const xd = randomNested(rng, [1, 2, 6]), wd = randomNested(rng, [2, 3, 3]);
    const oneD = flat(F.conv_transpose1d(tensor(xd), tensor(wd), null, 2, 1, 0, 1, 1));
    const twoD = flat(F.conv_transpose2d(
      tensor([xd[0].map(ch => [ch])]),
      tensor(wd.map(ci => ci.map(co => [co]))),
      null, [1, 2], [[0, 0], [1, 1]], [0, 0], [1, 1], 1,
    ));
    expect(maxRelErr(oneD, twoD)).toBeLessThan(1e-6);
  });
});

describe('ConvTranspose VJP matches finite differences', () => {
  const EPS = 1e-3, TOL = 5e-3;
  const CASES = [
    { name: 'stride 2 padding 1', xs: [1, 2, 4, 4], ws: [2, 3, 3, 3], stride: 2, padding: 1 },
    { name: 'stride 1 groups 2', xs: [1, 4, 4, 4], ws: [4, 2, 3, 3], stride: 1, padding: 1, groups: 2 },
  ];

  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 907);
      const shapes = [c.xs, c.ws];
      const datas = shapes.map((s) => randomNested(rng, s));
      const fwd = (xx, ww) => F.conv_transpose2d(xx, ww, null, c.stride, c.padding, 0, 1, c.groups ?? 1);

      const inputs = datas.map((d) => tensor(d));
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget() });
      const out = cf(...inputs);
      const analytic = cf.backward(ones(out.shape)).map(flat);

      for (let argi = 0; argi < 2; argi++) {
        const n = numel(shapes[argi]);
        const base = flat(tensor(datas[argi]));
        const step = Math.max(1, Math.floor(n / 8));
        for (let k = 0; k < n; k += step) {
          const sumAt = (delta) => {
            const arr = Array.from(base);
            arr[k] += delta;
            const args = datas.map((d, i) => tensor(i === argi ? nest(arr, shapes[argi]) : d));
            return flat(fwd(...args)).reduce((a, b) => a + b, 0);
          };
          const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
          const err = Math.abs(numeric - analytic[argi][k]) / (1 + Math.abs(numeric));
          expect(err, `arg${argi}[${k}]: numeric=${numeric} analytic=${analytic[argi][k]}`).toBeLessThan(TOL);
        }
      }
    });
  }
});

describe('ConvTranspose compiles to every backend', () => {
  const rng = mulberry32(31337);
  const build = () => {
    const m = new nn.ConvTranspose2d(2, 3, 3, { stride: 2, padding: 1, outputPadding: 1 });
    return { fwd: (x) => m.forward(x), input: tensor(randomNested(rng, [1, 2, 4, 4])) };
  };

  for (const [name, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`compiled on ${name} matches eager`, async () => {
      const { fwd, input } = build();
      const eager = flat(fwd(input));
      const compiled = compile({ forward: fwd }, [input], { target: makeTarget() });
      expect(maxRelErr(eager, flat(await compiled(input)))).toBeLessThan(2e-3);
    });
  }

  it('emits real CUDA source', () => {
    const { fwd, input } = build();
    const compiled = compile({ forward: fwd }, [input], { target: CUDATarget() });
    expect(compiled.source()).toMatch(/__global__\s+void/);
  });

  it('rejects padding that would crop the output below zero size', () => {
    const m = new nn.ConvTranspose2d(2, 3, 3, { padding: 5 });
    expect(() => m.forward(tensor(randomNested(mulberry32(1), [1, 2, 4, 4]))))
      .toThrow(/exceeds dilation\*\(kernel-1\)/);
  });
});
