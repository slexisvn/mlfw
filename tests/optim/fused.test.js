import { describe, it, expect } from 'vitest';
import { tensor, Parameter } from '../../src/index.js';
import { SGD } from '../../src/optim/sgd.js';
import { Adam } from '../../src/optim/adam.js';
import { FusedSGD, FusedAdam } from '../../src/optim/fused.js';

function mkParams() {
  return [
    new Parameter(tensor([[0.5, -0.3, 0.8], [0.1, 0.2, -0.6]])),
    new Parameter(tensor([0.4, -0.9, 0.7, 0.15])),
  ];
}

function setGrads(params, step) {
  for (let k = 0; k < params.length; k++) {
    const n = params[k]._impl.storage.data.length;
    const gd = new Float32Array(n);
    for (let i = 0; i < n; i++) gd[i] = Math.sin((i + 1) * 1.3 + step * 0.7 + k) * 0.2;
    params[k].grad = tensor(Array.from(gd));
  }
}

function runOptimizer(makeOpt, steps = 5) {
  const params = mkParams();
  const opt = makeOpt(params);
  for (let s = 0; s < steps; s++) {
    setGrads(params, s);
    opt.step();
  }
  return { params: params.map(p => Array.from(p._impl.storage.data)), opt };
}

function maxDiff(a, b) {
  let m = 0;
  for (let k = 0; k < a.length; k++) {
    for (let i = 0; i < a[k].length; i++) m = Math.max(m, Math.abs(a[k][i] - b[k][i]));
  }
  return m;
}

describe('FusedSGD matches eager SGD across configurations', () => {
  const CASES = [
    ['plain', { lr: 0.1 }],
    ['momentum', { lr: 0.1, momentum: 0.9 }],
    ['momentum+weightDecay', { lr: 0.1, momentum: 0.9, weightDecay: 0.01 }],
    ['nesterov', { lr: 0.1, momentum: 0.9, nesterov: true }],
    ['weightDecay only', { lr: 0.05, weightDecay: 0.02 }],
  ];
  for (const [name, opts] of CASES) {
    it(`${name}: fused == eager over 5 steps`, () => {
      const eager = runOptimizer(p => new SGD(p, opts)).params;
      const fused = runOptimizer(p => new FusedSGD(p, opts)).params;
      expect(maxDiff(eager, fused)).toBeLessThan(1e-5);
    });
  }
});

describe('FusedAdam matches eager Adam across configurations', () => {
  const CASES = [
    ['default', { lr: 0.01 }],
    ['weightDecay', { lr: 0.01, weightDecay: 0.02 }],
    ['amsgrad', { lr: 0.01, amsgrad: true }],
    ['weightDecay+amsgrad', { lr: 0.01, weightDecay: 0.02, amsgrad: true }],
  ];
  for (const [name, opts] of CASES) {
    it(`${name}: fused == eager over 5 steps`, () => {
      const eager = runOptimizer(p => new Adam(p, opts)).params;
      const fused = runOptimizer(p => new FusedAdam(p, opts)).params;
      expect(maxDiff(eager, fused)).toBeLessThan(1e-5);
    });
  }
});

describe('the compiled optimizer update is a single fused kernel', () => {
  it('SGD update fuses the whole elementwise chain into one loop with no temporaries', () => {
    const { opt } = runOptimizer(p => new FusedSGD(p, { lr: 0.1, momentum: 0.9, weightDecay: 0.01 }), 1);
    const kernels = [...opt._kernels.values()];
    expect(kernels.length).toBeGreaterThan(0);
    for (const k of kernels) {
      const src = k.getSource('sgd_update');
      expect((src.match(/for\s*\(/g) || []).length).toBe(1);
      expect((src.match(/new Float32Array/g) || []).length).toBe(0);
    }
  });

  it('Adam update (m, v, w multi-output) fuses into one loop with no temporaries', () => {
    const { opt } = runOptimizer(p => new FusedAdam(p, { lr: 0.01 }), 1);
    for (const k of opt._kernels.values()) {
      const src = k.getSource('adam_update');
      expect((src.match(/for\s*\(/g) || []).length).toBe(1);
      expect((src.match(/new Float32Array/g) || []).length).toBe(0);
    }
  });

  it('reuses one cached kernel per distinct parameter numel', () => {
    const params = [new Parameter(tensor([1, 2, 3])), new Parameter(tensor([4, 5, 6])), new Parameter(tensor([7, 8]))];
    for (const p of params) p.grad = tensor(p._impl.storage.data.map(() => 0.1));
    const opt = new FusedSGD(params, { lr: 0.1 });
    opt.step();
    expect(opt._kernels.size).toBe(2);
  });
});
