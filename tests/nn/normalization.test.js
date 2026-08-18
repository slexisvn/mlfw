import { describe, it, expect } from 'vitest';
import { tensor, compile, compileWithBackward, CPUTarget, WasmTarget, CUDATarget } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../_utils/tensor_data.js';

const F = nn.F;

const maxRelErr = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i]) / (1 + Math.abs(v))), 0);

function rmsNormReference(rows, eps) {
  return rows.flatMap((row) => {
    const ms = row.reduce((a, v) => a + v * v, 0) / row.length;
    const inv = 1 / Math.sqrt(ms + eps);
    return row.map((v) => v * inv);
  });
}

describe('RMSNorm matches a hand-computed reference', () => {
  const EPS = 1e-6;

  it('normalizes each row by its root-mean-square', () => {
    const rows = [[1, 2, 3, 4], [-2, 0.5, 0, 7], [0.1, 0.1, 0.1, 0.1]];
    const out = F.rms_norm(tensor(rows), [4], null, EPS);
    expect(maxRelErr(flat(out), rmsNormReference(rows, EPS))).toBeLessThan(1e-5);
  });

  it('leaves the mean alone, unlike LayerNorm', () => {
    const rows = [[1, 2, 3, 4]];
    const rms = flat(F.rms_norm(tensor(rows), [4], null, EPS));
    const ln = flat(F.layer_norm(tensor(rows), [4], null, null, EPS));
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    expect(Math.abs(mean(ln))).toBeLessThan(1e-5);
    expect(Math.abs(mean(rms))).toBeGreaterThan(0.1);
  });

  it('applies the affine weight elementwise', () => {
    const rows = [[1, 2, 3, 4]];
    const w = [2, 0.5, -1, 3];
    const plain = flat(F.rms_norm(tensor(rows), [4], null, EPS));
    const scaled = flat(F.rms_norm(tensor(rows), [4], tensor(w), EPS));
    expect(maxRelErr(scaled, plain.map((v, i) => v * w[i]))).toBeLessThan(1e-5);
  });

  it('the module defaults to a unit weight so it matches the functional form', () => {
    const m = new nn.RMSNorm(4);
    const rows = [[1, -2, 3, 0.5]];
    expect(maxRelErr(flat(m.forward(tensor(rows))), rmsNormReference(rows, m.eps))).toBeLessThan(1e-5);
  });

  it('normalizes over the trailing axes of a 3D input', () => {
    const rng = mulberry32(17);
    const data = randomNested(rng, [2, 3, 4]);
    const out = F.rms_norm(tensor(data), [4], null, EPS);
    expect(out.shape).toEqual([2, 3, 4]);
    const rows = data.flatMap((b) => b);
    expect(maxRelErr(flat(out), rmsNormReference(rows, EPS))).toBeLessThan(1e-5);
  });
});

describe('InstanceNorm normalizes each sample-channel plane independently', () => {
  it('drives every (n, c) plane to mean 0 and variance 1', () => {
    const rng = mulberry32(23);
    const shape = [2, 3, 4, 4];
    const out = flat(F.instance_norm(tensor(randomNested(rng, shape)), null, null, 1e-5));
    const plane = shape[2] * shape[3];
    for (let p = 0; p < shape[0] * shape[1]; p++) {
      const vals = out.slice(p * plane, (p + 1) * plane);
      const mean = vals.reduce((a, b) => a + b, 0) / plane;
      const variance = vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / plane;
      expect(Math.abs(mean), `plane ${p} mean`).toBeLessThan(1e-3);
      expect(Math.abs(variance - 1), `plane ${p} variance`).toBeLessThan(1e-2);
    }
  });

  it('equals GroupNorm with one group per channel', () => {
    const rng = mulberry32(29);
    const x = tensor(randomNested(rng, [2, 4, 3, 3]));
    const viaInstance = flat(F.instance_norm(x, null, null, 1e-5));
    const viaGroup = flat(F.group_norm(x, 4, null, null, 1e-5));
    expect(maxRelErr(viaInstance, viaGroup)).toBeLessThan(1e-6);
  });

  it('is unaffected by a per-sample shift, unlike BatchNorm', () => {
    const base = [[[[1, 2], [3, 4]]], [[[10, 20], [30, 40]]]];
    const out = flat(F.instance_norm(tensor(base), null, null, 1e-5));
    const first = out.slice(0, 4), second = out.slice(4, 8);
    expect(maxRelErr(first, second)).toBeLessThan(1e-4);
  });
});

describe('new norm layers differentiate correctly', () => {
  const EPS = 1e-3, TOL = 5e-3;
  const CASES = [
    { name: 'RMSNorm', shape: [2, 5], make: () => { const m = new nn.RMSNorm(5); return (x) => m.forward(x); } },
    { name: 'InstanceNorm2d', shape: [2, 3, 3, 3], make: () => { const m = new nn.InstanceNorm2d(3); return (x) => m.forward(x); } },
  ];

  for (const c of CASES) {
    it(`${c.name} VJP matches finite differences`, () => {
      const rng = mulberry32(c.name.length * 313);
      const data = randomNested(rng, c.shape);
      const fwd = c.make();
      const cf = compileWithBackward({ forward: fwd }, [tensor(data)], { target: CPUTarget() });
      const out = cf(tensor(data));
      const analytic = flat(cf.backward(ones(out.shape))[0]);

      const base = flat(tensor(data));
      const n = numel(c.shape);
      const step = Math.max(1, Math.floor(n / 8));
      for (let k = 0; k < n; k += step) {
        const sumAt = (delta) => {
          const arr = Array.from(base);
          arr[k] += delta;
          return flat(fwd(tensor(nest(arr, c.shape)))).reduce((a, b) => a + b, 0);
        };
        const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
        expect(Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric)), `${c.name}[${k}]`).toBeLessThan(TOL);
      }
    });
  }
});

describe('new norm layers compile to every backend', () => {
  const CASES = [
    { name: 'RMSNorm', shape: [2, 3, 8], make: () => { const m = new nn.RMSNorm(8); return (x) => m.forward(x); } },
    { name: 'InstanceNorm2d', shape: [2, 4, 4, 4], make: () => { const m = new nn.InstanceNorm2d(4); return (x) => m.forward(x); } },
  ];

  for (const c of CASES) {
    for (const [name, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
      it(`${c.name} compiled on ${name} matches eager`, async () => {
        const input = tensor(randomNested(mulberry32(c.name.length * 7), c.shape));
        const fwd = c.make();
        const eager = flat(fwd(input));
        const compiled = compile({ forward: fwd }, [input], { target: makeTarget() });
        expect(maxRelErr(eager, flat(await compiled(input)))).toBeLessThan(2e-3);
      });
    }

    it(`${c.name} emits real CUDA source`, () => {
      const input = tensor(randomNested(mulberry32(5), c.shape));
      expect(compile({ forward: c.make() }, [input], { target: CUDATarget() }).source()).toMatch(/__global__\s+void/);
    });
  }
});
