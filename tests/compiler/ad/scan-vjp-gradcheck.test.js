import { describe, it, expect } from 'vitest';
import { tensor } from '../../../src/index.js';
import * as nn from '../../../src/nn/index.js';
import { ones } from '../../../src/tensor/factory/creation_ops.js';
import { compileWithBackward } from '../../../src/tracing/compile_backward.js';
import { CPUTarget } from '../../../src/compiler/support/target.js';
import { mulberry32 } from '../../_utils/rng.js';
import { randomNested, flat, numel, nest } from '../../_utils/tensor_data.js';
import { buildFunction, IRBuilder } from '../../../src/compiler/ir/graph/builder.js';
import { TensorType } from '../../../src/compiler/ir/graph/types.js';
import { BackwardGraphBuilder } from '../../../src/compiler/ad/backward_builder.js';
import { compileGraph } from '../../../src/compiler/pipeline/compiler.js';

const EPS = 2e-3;
const TOL = 5e-3;

describe('scan VJP with explicit constant operands', () => {
  it('accumulates each constant gradient across timesteps and keeps carry/xs gradients in order', () => {
    const carryType = new TensorType([2], 'f32');
    const xsType = new TensorType([3, 2], 'f32');
    const forward = buildFunction('weighted_scan', [carryType, xsType, carryType], [carryType, xsType], (b, [carry, xs, weight]) => {
      const scan = b.scanOp([carry], [xs], (bb, c, x, k) => {
        const next = bb.add(c[0], bb.mul(x[0], k[0]).getResult(0)).getResult(0);
        return [[next], [next]];
      }, [weight]);
      b.returnOp(scan.results);
    });
    const { backwardFunc, savedValues, gradInputIndices } = new BackwardGraphBuilder().build(forward);
    const outputs = [...forward.getReturnOp().operands, ...savedValues];
    forward.getReturnOp().erase();
    new IRBuilder(forward).returnOp(outputs);
    forward.outputTypes = Object.freeze(outputs.map((v) => v.type));
    const savedBuffers = outputs.map((v) => new Float32Array(numel(v.type.shape)));
    compileGraph(forward, CPUTarget()).run('weighted_scan',
      new Float32Array([10, 20]), new Float32Array([1, 2, 3, 4, 5, 6]), new Float32Array([2, -1]), ...savedBuffers);
    expect([...savedBuffers[0]]).toEqual([28, 8]);
    expect([...savedBuffers[1]]).toEqual([12, 18, 18, 14, 28, 8]);
    const gradients = backwardFunc.outputTypes.map((type) => new Float32Array(numel(type.shape)));
    compileGraph(backwardFunc, CPUTarget()).run(backwardFunc.name,
      new Float32Array([2, 3]), new Float32Array([1, -2, 3, -4, 5, -6]), ...savedBuffers.slice(2), ...gradients);
    const byInput = new Map(gradInputIndices.map((index, i) => [index, [...gradients[i]]]));
    expect(byInput.get(0)).toEqual([11, -9]);
    expect(byInput.get(1)).toEqual([22, 9, 20, 7, 14, 3]);
    expect(byInput.get(2)).toEqual([76, -64]);
  });
});

const CASES = [
  { name: 'LSTM single layer', make: () => { const m = new nn.LSTM(3, 4, 1, true); return (x) => m.forward(x)[0]; }, shape: [1, 4, 3] },
  { name: 'GRU single layer', make: () => { const m = new nn.GRU(3, 4, 1, true); return (x) => m.forward(x)[0]; }, shape: [1, 4, 3] },
  { name: 'LSTM two layers', make: () => { const m = new nn.LSTM(3, 4, 2, true); return (x) => m.forward(x)[0]; }, shape: [1, 3, 3] },
];

describe.each(['separate', 'joint'])('scan VJP (%s mode) matches finite differences', (mode) => {
  for (const c of CASES) {
    it(c.name, () => {
      const rng = mulberry32(c.name.length * 13 + 1);
      const data = randomNested(rng, c.shape);
      const fwd = c.make();

      const inputs = [tensor(data)];
      const cf = compileWithBackward({ forward: (x) => fwd(x) }, inputs, { target: CPUTarget(), mode });
      const out = cf(...inputs);
      const analytic = flat(cf.backward(ones(out.shape))[0]);

      const base = flat(tensor(data));
      const n = numel(c.shape);
      expect(analytic.length).toBe(n);

      for (let k = 0; k < n; k++) {
        const sumAt = (delta) => {
          const arr = Array.from(base);
          arr[k] += delta;
          return flat(fwd(tensor(nest(arr, c.shape)))).reduce((a, b) => a + b, 0);
        };
        const numeric = (sumAt(EPS) - sumAt(-EPS)) / (2 * EPS);
        const relErr = Math.abs(numeric - analytic[k]) / (1 + Math.abs(numeric));
        expect(relErr, `${c.name}[${k}]: numeric=${numeric} analytic=${analytic[k]}`).toBeLessThan(TOL);
      }
    }, 120000);
  }
});
