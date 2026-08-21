import { describe, it, expect } from 'vitest';
import { tensor, matmul, relu, tanh, sum, mean } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { compileWithBackward } from '../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../src/backend/target.js';
import { ones, zeros } from '../../src/tensor/factory/creation_ops.js';
import { mulberry32 } from '../_utils/rng.js';
import { randomTensor, flat, nest, numel } from '../_utils/tensor_data.js';

const MODES = ['separate', 'joint'];

const flatAll = (tensors) => tensors.flatMap((t) => flat(t));

function multiHead() {
  const trunk = new nn.Linear(12, 16), a = new nn.Linear(16, 4), b = new nn.Linear(16, 3);
  return {
    fwd: (x) => { const h = relu(trunk.forward(x)); return [a.forward(h), b.forward(h)]; },
    inputs: [randomTensor(mulberry32(5), [4, 12])],
  };
}

describe('compiled backward validates its gradient arguments', () => {
  for (const mode of MODES) {
    it(`${mode}: rejects too few gradients with a named error`, async () => {
      const { fwd, inputs } = multiHead();
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      expect(() => cf.backward(ones(out[0].shape)))
        .toThrow(/one gradient per forward output.*returns 2 output\(s\) but 1 gradient\(s\)/);
    });

    it(`${mode}: rejects too many gradients`, async () => {
      const { fwd, inputs } = multiHead();
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      const extra = [...out.map((t) => ones(t.shape)), ones([2, 2])];
      expect(() => cf.backward(...extra)).toThrow(/one gradient per forward output/);
    });

    it(`${mode}: rejects a gradient whose element count does not match its output`, async () => {
      const { fwd, inputs } = multiHead();
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      expect(() => cf.backward(ones(out[0].shape), zeros([9, 9])))
        .toThrow(/gradient 1 has 81 element\(s\) but forward output 1 has 12/);
    });

    it(`${mode}: accepts one gradient per output and differentiates every head`, async () => {
      const { fwd, inputs } = multiHead();
      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      let grads = cf.backward(...out.map((t) => ones(t.shape)));
      if (grads && grads.then) grads = await grads;
      const values = flatAll(grads);
      expect(values.length).toBeGreaterThan(0);
      expect(values.every(Number.isFinite)).toBe(true);
      expect(values.some((v) => v !== 0)).toBe(true);
    });
  }
});

describe('scan-based recurrent models differentiate through compiled backward', () => {
  for (const mode of MODES) {
    it(`${mode}: LSTM returning output plus final states`, async () => {
      const lstm = new nn.LSTM(8, 12, 2, true); lstm.eval();
      const fwd = (x) => { const [o, [h, c]] = lstm.forward(x); return [o, h, c]; };
      const inputs = [randomTensor(mulberry32(20), [3, 5, 8])];

      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      expect(out.length).toBe(3);

      let grads = cf.backward(...out.map((t) => ones(t.shape)));
      if (grads && grads.then) grads = await grads;
      const values = flatAll(grads);
      expect(values.every(Number.isFinite), 'scan gradients must be finite').toBe(true);
      expect(values.some((v) => v !== 0), 'scan gradients must not be all zero').toBe(true);
    });

    it(`${mode}: GRU single output`, async () => {
      const gru = new nn.GRU(8, 12, 1, true); gru.eval();
      const fwd = (x) => { const [o] = gru.forward(x); return o; };
      const inputs = [randomTensor(mulberry32(21), [3, 5, 8])];

      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      const first = Array.isArray(out) ? out[0] : out;

      let grads = cf.backward(ones(first.shape));
      if (grads && grads.then) grads = await grads;
      const values = flatAll(grads);
      expect(values.every(Number.isFinite)).toBe(true);
      expect(values.some((v) => v !== 0)).toBe(true);
    });
  }
});

describe('compiled backward gradients agree with finite differences', () => {
  const SHAPES = [[4, 6], [6, 3]];
  const fwd = (x, w) => mean(tanh(matmul(x, w)));

  function inputsFrom(flatData) {
    return SHAPES.map((shape, i) => tensor(nest(flatData[i], shape)));
  }

  for (const mode of MODES) {
    it(`${mode}: mean(tanh(x @ w))`, async () => {
      const rng = mulberry32(31);
      const flatData = SHAPES.map((shape) => Array.from({ length: numel(shape) }, () => rng() * 2 - 1));
      const inputs = inputsFrom(flatData);

      const cf = compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      const scalar = Array.isArray(out) ? out[0] : out;
      let grads = cf.backward(ones(scalar.shape));
      if (grads && grads.then) grads = await grads;

      const eps = 1e-3;
      for (let arg = 0; arg < SHAPES.length; arg++) {
        const analytic = flat(grads[arg]);
        expect(analytic.length).toBe(numel(SHAPES[arg]));
        for (let k = 0; k < analytic.length; k += 5) {
          const plus = flatData.map((d) => d.slice());
          const minus = flatData.map((d) => d.slice());
          plus[arg][k] += eps;
          minus[arg][k] -= eps;
          const numeric = (flat(fwd(...inputsFrom(plus)))[0] - flat(fwd(...inputsFrom(minus)))[0]) / (2 * eps);
          expect(Math.abs(analytic[k] - numeric), `arg ${arg} idx ${k}: analytic=${analytic[k]} numeric=${numeric}`)
            .toBeLessThan(2e-3);
        }
      }
    });
  }
});

describe('joint mode runs its single kernel with the real cotangent', () => {
  async function settle(v) { return v && v.then ? await v : v; }

  const trunk = new nn.Linear(12, 16), headA = new nn.Linear(16, 4), headB = new nn.Linear(16, 3);
  const fwd = (x) => { const h = relu(trunk.forward(x)); return [headA.forward(h), headB.forward(h)]; };
  const inputs = [randomTensor(mulberry32(5), [4, 12])];
  const OUT_SHAPES = [[4, 4], [4, 3]];

  function compiled(mode) {
    return compileWithBackward({ forward: fwd }, inputs, { target: CPUTarget(), mode });
  }

  async function separateReference() {
    const cf = compiled('separate');
    const out = await settle(cf(...inputs));
    const grads = await settle(cf.backward(...OUT_SHAPES.map((s) => ones(s))));
    return { outputs: flatAll(out), gradients: flatAll(grads) };
  }

  it('forward outputs match separate mode when backward is never called', async () => {
    const reference = await separateReference();
    expect(flatAll(await settle(compiled('joint')(...inputs)))).toEqual(reference.outputs);
  });

  it('forward outputs stay correct when they are read only after backward', async () => {
    const reference = await separateReference();
    const cf = compiled('joint');
    const out = await settle(cf(...inputs));
    await settle(cf.backward(...OUT_SHAPES.map((s) => ones(s))));
    expect(flatAll(out)).toEqual(reference.outputs);
  });

  it('gradients match separate mode whether or not the outputs were read first', async () => {
    const reference = await separateReference();

    const unread = compiled('joint');
    const unreadOut = await settle(unread(...inputs));
    const unreadGrads = await settle(unread.backward(...OUT_SHAPES.map((s) => ones(s))));

    const read = compiled('joint');
    const readOut = await settle(read(...inputs));
    flatAll(readOut);
    const readGrads = await settle(read.backward(...OUT_SHAPES.map((s) => ones(s))));

    expect(flatAll(unreadGrads)).toEqual(reference.gradients);
    expect(flatAll(readGrads)).toEqual(reference.gradients);
    expect(flatAll(unreadOut)).toEqual(reference.outputs);
  });

  it('repeated backward calls return independent gradient tensors', async () => {
    const cf = compiled('joint');
    const out = await settle(cf(...inputs));
    const shapes = out.map((t) => t.shape);

    const first = await settle(cf.backward(...shapes.map((s) => ones(s))));
    const firstValues = flatAll(first);
    const second = await settle(cf.backward(...shapes.map((s) => zeros(s))));

    expect(flatAll(second).every((v) => v === 0)).toBe(true);
    expect(flatAll(first)).toEqual(firstValues);
    expect(firstValues.some((v) => v !== 0)).toBe(true);
  });
});

describe('backward before forward is rejected', () => {
  it('names the missing forward pass', () => {
    const { fwd } = multiHead();
    const cf = compileWithBackward({ forward: fwd }, undefined, { target: CPUTarget() });
    expect(() => cf.backward(ones([1]))).toThrow(/Must run forward before backward/);
  });
});

describe('sum reduction keeps gradient shapes aligned with inputs', () => {
  for (const mode of MODES) {
    it(`${mode}: gradient of sum has the input's element count`, async () => {
      const inputs = [randomTensor(mulberry32(41), [5, 7])];
      const cf = compileWithBackward({ forward: (x) => sum(x) }, inputs, { target: CPUTarget(), mode });
      let out = cf(...inputs); if (out && out.then) out = await out;
      const scalar = Array.isArray(out) ? out[0] : out;
      let grads = cf.backward(ones(scalar.shape));
      if (grads && grads.then) grads = await grads;
      expect(flat(grads[0]).length).toBe(35);
      expect(flat(grads[0]).every((v) => Math.abs(v - 1) < 1e-6)).toBe(true);
    });
  }
});
