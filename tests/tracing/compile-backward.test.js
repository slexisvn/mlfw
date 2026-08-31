import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import * as nn from '../../src/nn/index.js';
import { log_softmax, sigmoid } from '../../src/nn/functional/activation.js';
import { mse_loss, nll_loss, cross_entropy, binary_cross_entropy } from '../../src/nn/functional/loss.js';
import { compileWithBackward } from '../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../src/compiler/support/target.js';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import { flat } from '../_utils/tensor_data.js';

const MODES = ['separate', 'joint'];

function grid(rows, cols, phase) {
  const data = [];
  for (let i = 0; i < rows; i++) {
    const r = [];
    for (let d = 0; d < cols; d++) r.push(Math.sin(i * 0.5 + d * 0.3 + phase));
    data.push(r);
  }
  return tensor(data);
}

function activationHeads() {
  const mods = [
    new nn.LeakyReLU(0.1), new nn.ELU(1.0), new nn.Softmax(-1), new nn.LogSoftmax(-1),
    new nn.Sigmoid(), new nn.Tanh(), new nn.GELU(), new nn.SiLU(), new nn.ReLU(),
  ];
  return {
    model: { forward: (x) => mods.map((m) => m.forward(x)) },
    input: grid(4, 6, 0),
  };
}

function lossHeads() {
  const lin = new nn.Linear(12, 5);
  lin.eval();
  const classes = tensor([0, 1, 2, 3, 4, 0]);
  const dense = grid(6, 5, 1.7).sigmoid();
  return {
    model: {
      forward: (x) => {
        const logits = lin.forward(x);
        return [
          cross_entropy(logits, classes),
          mse_loss(logits, dense),
          nll_loss(log_softmax(logits, -1), classes),
          binary_cross_entropy(sigmoid(logits), dense),
        ];
      },
    },
    input: grid(6, 12, 0.4),
  };
}

async function gradientsFor(model, input, mode) {
  const cf = compileWithBackward(model, [input], { target: CPUTarget(), mode });
  let out = cf(input);
  if (out && out.then) out = await out;
  const outputs = Array.isArray(out) ? out : [out];
  let grads = cf.backward(...outputs.map((t) => ones(t.shape)));
  if (grads && grads.then) grads = await grads;
  return { grads, numForwardInputs: 1 + cf.capturedParams().length };
}

describe('compiled backward returns the same gradient set in both modes', () => {
  const cases = [
    { name: 'nine activation heads over one [4,6] input', build: activationHeads },
    { name: 'four loss heads over Linear(12,5) on a [6,12] input', build: lossHeads },
  ];

  for (const { name, build } of cases) {
    it(`${name}: modes agree on gradient count and values`, async () => {
      const { model, input } = build();

      const separate = await gradientsFor(model, input, 'separate');
      const joint = await gradientsFor(model, input, 'joint');

      expect(separate.grads.length, 'separate returns one gradient per forward input')
        .toBe(separate.numForwardInputs);
      expect(joint.grads.length, 'joint returns one gradient per forward input')
        .toBe(joint.numForwardInputs);
      expect(separate.grads.length).toBe(joint.grads.length);

      for (let i = 0; i < separate.grads.length; i++) {
        expect(separate.grads[i].shape, `gradient ${i} shape`).toEqual(joint.grads[i].shape);
        const a = flat(separate.grads[i]);
        const b = flat(joint.grads[i]);
        expect(a.length).toBe(b.length);
        for (let k = 0; k < a.length; k++) {
          expect(Number.isFinite(a[k]), `separate gradient ${i}[${k}] is finite`).toBe(true);
          expect(a[k], `gradient ${i}[${k}]`).toBeCloseTo(b[k], 6);
        }
      }

      const values = separate.grads.flatMap((t) => flat(t));
      expect(values.some((v) => v !== 0), 'the gradient set is not uniformly zero').toBe(true);
    }, 30000);

    it(`${name}: keeps a zero slot for inputs with no differentiable path`, async () => {
      const { model, input } = build();

      const perMode = {};
      for (const mode of MODES) {
        const { grads } = await gradientsFor(model, input, mode);
        perMode[mode] = grads.map((t) => flat(t).reduce((acc, v) => acc + Math.abs(v), 0));
      }

      const zeroSlots = (sums) => sums.map((s, i) => (s === 0 ? i : -1)).filter((i) => i >= 0);
      expect(zeroSlots(perMode.separate).length, 'this model has a structurally zero gradient')
        .toBeGreaterThan(0);
      expect(zeroSlots(perMode.separate), 'both modes place the zero gradients at the same indices')
        .toEqual(zeroSlots(perMode.joint));
    }, 30000);
  }
});
