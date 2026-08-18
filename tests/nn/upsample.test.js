import { describe, it, expect } from 'vitest';
import { tensor, compile, compileWithBackward, CPUTarget, WasmTarget, CUDATarget } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../_utils/tensor_data.js';

const F = nn.F;

function nearestReference(data, shape, scales) {
  const outShape = shape.map((d, i) => (i < 2 ? d : d * scales[i - 2]));
  const strideOf = (s) => s.reduceRight((acc, d) => { acc.unshift((acc[0] ?? 1) * d); return acc; }, []).slice(1).concat([1]);
  const inStride = strideOf(shape), outStride = strideOf(outShape);
  const out = new Array(outShape.reduce((a, b) => a * b, 1)).fill(0);
  const idx = new Array(outShape.length).fill(0);
  for (let o = 0; o < out.length; o++) {
    let rem = o;
    for (let d = 0; d < outShape.length; d++) { idx[d] = Math.floor(rem / outStride[d]); rem %= outStride[d]; }
    let src = 0;
    for (let d = 0; d < shape.length; d++) {
      const coord = d < 2 ? idx[d] : Math.floor(idx[d] / scales[d - 2]);
      src += coord * inStride[d];
    }
    out[o] = data[src];
  }
  return out;
}

describe('nearest interpolate matches an independent reference', () => {
  const CASES = [
    { shape: [1, 1, 2, 2], scales: [2, 2] },
    { shape: [2, 3, 3, 4], scales: [2, 3] },
    { shape: [1, 2, 4, 4], scales: [3, 1] },
    { shape: [2, 2, 5], scales: [4] },
    { shape: [1, 2, 2, 2, 2], scales: [2, 2, 2] },
  ];

  for (const c of CASES) {
    it(`shape [${c.shape}] by [${c.scales}]`, () => {
      const rng = mulberry32(c.shape.length * 71 + c.scales[0]);
      const data = randomNested(rng, c.shape);
      const x = tensor(data);
      const out = F.interpolate(x, { scaleFactor: c.scales.length === 1 ? c.scales[0] : c.scales });
      const expectedShape = c.shape.map((d, i) => (i < 2 ? d : d * c.scales[i - 2]));
      expect(out.shape).toEqual(expectedShape);
      expect(flat(out)).toEqual(nearestReference(flat(x), c.shape, c.scales));
    });
  }

  it('a scale factor of 1 returns the input unchanged', () => {
    const x = tensor([[[[1, 2], [3, 4]]]]);
    expect(flat(F.interpolate(x, { scaleFactor: 1 }))).toEqual(flat(x));
  });

  it('target size is accepted when it is an integer multiple', () => {
    const x = tensor(randomNested(mulberry32(3), [1, 2, 2, 3]));
    const out = F.interpolate(x, { size: [6, 9] });
    expect(out.shape).toEqual([1, 2, 6, 9]);
  });

  it('rejects a non-integer upsampling factor', () => {
    const x = tensor(randomNested(mulberry32(4), [1, 1, 3, 3]));
    expect(() => F.interpolate(x, { size: [4, 4] })).toThrow(/integer factor/);
    expect(() => F.interpolate(x, { scaleFactor: 1.5 })).toThrow(/integer scale factors/);
  });

  it('rejects passing both size and scaleFactor, or neither', () => {
    const x = tensor(randomNested(mulberry32(5), [1, 1, 2, 2]));
    expect(() => F.interpolate(x, { size: [4, 4], scaleFactor: 2 })).toThrow(/exactly one/);
    expect(() => F.interpolate(x, {})).toThrow(/exactly one/);
  });

  it('rejects an unimplemented mode instead of silently doing nearest', () => {
    const x = tensor(randomNested(mulberry32(6), [1, 1, 2, 2]));
    expect(() => F.interpolate(x, { scaleFactor: 2, mode: 'bilinear' })).toThrow(/unsupported mode/);
  });
});

describe('Upsample VJP matches finite differences', () => {
  const EPS = 1e-3, TOL = 5e-3;

  it('gradient sums the contributions of every replicated output cell', () => {
    const shape = [1, 2, 3, 3];
    const rng = mulberry32(818);
    const data = randomNested(rng, shape);
    const up = new nn.Upsample({ scaleFactor: [2, 3] });
    const fwd = (x) => up.forward(x);

    const cf = compileWithBackward({ forward: fwd }, [tensor(data)], { target: CPUTarget() });
    const out = cf(tensor(data));
    const analytic = flat(cf.backward(ones(out.shape))[0]);

    const base = flat(tensor(data));
    const n = numel(shape);
    for (let k = 0; k < n; k++) {
      const sumAt = (delta) => {
        const arr = Array.from(base);
        arr[k] += delta;
        return flat(fwd(tensor(nest(arr, shape)))).reduce((a, b) => a + b, 0);
      };
      const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
      expect(Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric)), `[${k}]`).toBeLessThan(TOL);
      expect(analytic[k]).toBeCloseTo(6, 3);
    }
  });
});

describe('Upsample compiles to every backend', () => {
  const build = () => {
    const up = new nn.Upsample({ scaleFactor: 2 });
    return { fwd: (x) => up.forward(x), input: tensor(randomNested(mulberry32(99), [1, 2, 3, 3])) };
  };

  for (const [name, makeTarget] of [['cpu', CPUTarget], ['wasm', WasmTarget]]) {
    it(`compiled on ${name} matches eager`, async () => {
      const { fwd, input } = build();
      const eager = flat(fwd(input));
      const compiled = compile({ forward: fwd }, [input], { target: makeTarget() });
      expect(flat(await compiled(input))).toEqual(eager);
    });
  }

  it('emits real CUDA source', () => {
    const { fwd, input } = build();
    expect(compile({ forward: fwd }, [input], { target: CUDATarget() }).source()).toMatch(/__global__\s+void/);
  });
});
