import { describe, it, expect } from 'vitest';
import { tensor, compile, compileWithBackward, CPUTarget, WasmTarget, CUDATarget } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../_utils/tensor_data.js';

const F = nn.F;
const maxRelErr = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]) / (1 + Math.abs(v))), 0);

function conv3dReference(x, w, shapeX, shapeW, stride, padding, dilation, groups) {
  const [N, Ci, D, H, W] = shapeX;
  const [Co, CiPerGroup, kD, kH, kW] = shapeW;
  const [sD, sH, sW] = stride, [pD, pH, pW] = padding, [dD, dH, dW] = dilation;
  const outD = Math.floor((D + 2 * pD - dD * (kD - 1) - 1) / sD) + 1;
  const outH = Math.floor((H + 2 * pH - dH * (kH - 1) - 1) / sH) + 1;
  const outW = Math.floor((W + 2 * pW - dW * (kW - 1) - 1) / sW) + 1;
  const coPerGroup = Co / groups;
  const at = (n, c, d, h, ww) => x[((((n * Ci + c) * D + d) * H + h) * W) + ww];
  const wAt = (co, ci, kd, kh, kw) => w[((((co * CiPerGroup + ci) * kD + kd) * kH + kh) * kW) + kw];

  const out = [];
  for (let n = 0; n < N; n++) {
    for (let co = 0; co < Co; co++) {
      const g = Math.floor(co / coPerGroup);
      for (let od = 0; od < outD; od++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let acc = 0;
            for (let ci = 0; ci < CiPerGroup; ci++) {
              for (let kd = 0; kd < kD; kd++) {
                const id = od * sD - pD + kd * dD;
                if (id < 0 || id >= D) continue;
                for (let kh = 0; kh < kH; kh++) {
                  const ih = oh * sH - pH + kh * dH;
                  if (ih < 0 || ih >= H) continue;
                  for (let kw = 0; kw < kW; kw++) {
                    const iw = ow * sW - pW + kw * dW;
                    if (iw < 0 || iw >= W) continue;
                    acc += at(n, g * CiPerGroup + ci, id, ih, iw) * wAt(co, ci, kd, kh, kw);
                  }
                }
              }
            }
            out.push(acc);
          }
        }
      }
    }
  }
  return out;
}

describe('conv3d matches a direct-summation reference (independent oracle)', () => {
  const CASES = [
    { name: 'basic', x: [1, 2, 4, 4, 4], w: [3, 2, 3, 3, 3], stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1], groups: 1 },
    { name: 'padded', x: [1, 2, 4, 4, 4], w: [3, 2, 3, 3, 3], stride: [1, 1, 1], padding: [1, 1, 1], dilation: [1, 1, 1], groups: 1 },
    { name: 'strided depth', x: [2, 2, 6, 5, 5], w: [3, 2, 3, 3, 3], stride: [2, 1, 1], padding: [1, 1, 1], dilation: [1, 1, 1], groups: 1 },
    { name: 'strided all', x: [1, 2, 6, 6, 6], w: [2, 2, 3, 3, 3], stride: [2, 2, 2], padding: [1, 1, 1], dilation: [1, 1, 1], groups: 1 },
    { name: 'anisotropic kernel', x: [1, 2, 5, 6, 7], w: [3, 2, 1, 3, 5], stride: [1, 1, 1], padding: [0, 1, 2], dilation: [1, 1, 1], groups: 1 },
    { name: 'dilated depth', x: [1, 2, 7, 5, 5], w: [2, 2, 3, 3, 3], stride: [1, 1, 1], padding: [2, 1, 1], dilation: [2, 1, 1], groups: 1 },
    { name: 'groups 2', x: [1, 4, 4, 4, 4], w: [4, 2, 3, 3, 3], stride: [1, 1, 1], padding: [1, 1, 1], dilation: [1, 1, 1], groups: 2 },
  ];

  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 613 + c.x[2]);
      const xd = randomNested(rng, c.x), wd = randomNested(rng, c.w);
      const out = F.conv3d(tensor(xd), tensor(wd), null, c.stride, c.padding, c.dilation, c.groups);
      const reference = conv3dReference(flat(tensor(xd)), flat(tensor(wd)), c.x, c.w, c.stride, c.padding, c.dilation, c.groups);
      expect(flat(out).length, 'output element count').toBe(reference.length);
      expect(maxRelErr(flat(out), reference)).toBeLessThan(1e-5);
    });
  }

  it('a depth-1 kernel on a depth-1 input equals conv2d', () => {
    const rng = mulberry32(555);
    const xd = randomNested(rng, [1, 2, 1, 5, 5]), wd = randomNested(rng, [3, 2, 1, 3, 3]);
    const threeD = flat(F.conv3d(tensor(xd), tensor(wd), null, 1, [0, 1, 1], 1, 1));
    const twoD = flat(F.conv2d(tensor(xd[0].map(c => c[0])).reshape([1, 2, 5, 5]), tensor(wd.map(co => co.map(ci => ci[0]))), null, 1, 1, 1, 1));
    expect(maxRelErr(threeD, twoD)).toBeLessThan(1e-6);
  });

  it('the module adds bias per output channel', () => {
    const m = new nn.Conv3d(2, 3, 3, { padding: 1 });
    const rng = mulberry32(77);
    const x = tensor(randomNested(rng, [1, 2, 4, 4, 4]));
    const withBias = flat(m.forward(x));
    const noBias = flat(F.conv3d(x, m.weight, null, m.stride, m.padding, m.dilation, m.groups));
    const b = flat(m.bias);
    const plane = 4 * 4 * 4;
    for (let co = 0; co < 3; co++) {
      for (let i = 0; i < plane; i++) {
        expect(withBias[co * plane + i]).toBeCloseTo(noBias[co * plane + i] + b[co], 4);
      }
    }
  });
});

describe('conv3d VJP matches finite differences', () => {
  const EPS = 1e-3, TOL = 5e-3;
  const CASES = [
    { name: 'stride 1 padded', xs: [1, 2, 4, 4, 4], ws: [2, 2, 3, 3, 3], stride: 1, padding: 1, groups: 1 },
    { name: 'strided depth, groups 2', xs: [1, 4, 5, 4, 4], ws: [4, 2, 3, 3, 3], stride: [2, 1, 1], padding: 1, groups: 2 },
  ];

  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 401);
      const shapes = [c.xs, c.ws];
      const datas = shapes.map((s) => randomNested(rng, s));
      const fwd = (xx, ww) => F.conv3d(xx, ww, null, c.stride, c.padding, 1, c.groups);

      const inputs = datas.map((d) => tensor(d));
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget() });
      const out = cf(...inputs);
      const analytic = cf.backward(ones(out.shape)).map(flat);

      for (let argi = 0; argi < 2; argi++) {
        const n = numel(shapes[argi]);
        const base = flat(tensor(datas[argi]));
        const step = Math.max(1, Math.floor(n / 6));
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

describe('conv3d compiles to every backend', () => {
  const build = () => {
    const m = new nn.Conv3d(2, 3, 3, { padding: 1 });
    return { fwd: (x) => m.forward(x), input: tensor(randomNested(mulberry32(2024), [1, 2, 4, 4, 4])) };
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
    expect(compile({ forward: fwd }, [input], { target: CUDATarget() }).source()).toMatch(/__global__\s+void/);
  });
});
