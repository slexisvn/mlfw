import { describe, it, expect } from 'vitest';
import { tensor, Parameter, matmul, add, sub, mean, mul } from '../../src/index.js';
import { Adam } from '../../src/optim/adam.js';

describe('Adam single step correctness', () => {
  it('matches hand-computed update with bias correction', () => {
    const p = new Parameter(tensor([0.5]));
    p.grad = tensor([2.0]);
    const lr = 0.001;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const opt = new Adam([p], { lr, betas: [beta1, beta2], eps });

    opt.step();

    const m = (1 - beta1) * 2.0;
    const v = (1 - beta2) * 4.0;
    const mHat = m / (1 - beta1);
    const vHat = v / (1 - beta2);
    const expected = 0.5 - lr * mHat / (Math.sqrt(vHat) + eps);
    expect(p._impl.storage.data[0]).toBeCloseTo(expected, 6);
  });
});

describe('Adam multi-step', () => {
  it('state tracks step count and moment estimates', () => {
    const p = new Parameter(tensor([1.0, 2.0]));
    p.grad = tensor([0.5, -0.5]);
    const opt = new Adam([p], { lr: 0.01 });

    opt.step();
    const state = opt._getState(p);
    expect(state.step).toBe(1);
    expect(state.expAvg).toBeInstanceOf(Float32Array);
    expect(state.expAvgSq).toBeInstanceOf(Float32Array);

    p.grad = tensor([0.3, -0.7]);
    opt.step();
    expect(state.step).toBe(2);
  });
});

describe('Adam with weight decay', () => {
  it('adds L2 penalty to gradient', () => {
    const p1 = new Parameter(tensor([5.0]));
    const p2 = new Parameter(tensor([5.0]));

    const noWd = new Adam([p1], { lr: 0.1, weightDecay: 0 });
    const withWd = new Adam([p2], { lr: 0.1, weightDecay: 1.0 });
    for (let i = 0; i < 10; i++) {
      p1.grad = tensor([1.0]);
      p2.grad = tensor([1.0]);
      noWd.step();
      withWd.step();
    }
    const diff = Math.abs(p1._impl.storage.data[0] - p2._impl.storage.data[0]);
    expect(diff).toBeGreaterThan(0.001);
  });
});

describe('Adam AMSGrad', () => {
  it('maxExpAvgSq is monotonically non-decreasing', () => {
    const p = new Parameter(tensor([1.0]));
    const opt = new Adam([p], { lr: 0.01, amsgrad: true });

    const gradValues = [10.0, 0.01, 5.0, 0.001];
    let prevMax = 0;
    for (const gv of gradValues) {
      p.grad = tensor([gv]);
      opt.step();
      const state = opt._getState(p);
      expect(state.maxExpAvgSq[0]).toBeGreaterThanOrEqual(prevMax);
      prevMax = state.maxExpAvgSq[0];
    }
  });
});

describe('Adam stateDict round-trip', () => {
  it('preserves moments across save/load', () => {
    const p = new Parameter(tensor([1.0, 2.0, 3.0]));
    p.grad = tensor([0.1, 0.2, 0.3]);
    const opt1 = new Adam([p], { lr: 0.01 });
    for (let i = 0; i < 5; i++) {
      opt1.step();
      p.grad = tensor([0.1 * (i + 2), 0.2 * (i + 2), 0.3 * (i + 2)]);
    }
    const dict = opt1.stateDict();

    const p2 = new Parameter(tensor([1.0, 2.0, 3.0]));
    const opt2 = new Adam([p2], { lr: 0.99 });
    opt2.loadStateDict(dict);
    expect(opt2.paramGroups[0].lr).toBe(0.01);

    const s1 = opt1._getState(p);
    const s2 = opt2._state.get(opt2._getParamId(p2));
    expect(s2.step).toBe(s1.step);
    expect([...s2.expAvg]).toEqual([...s1.expAvg]);
  });
});

describe('Adam convergence', () => {
  it('decreases loss on regression', () => {
    const w = new Parameter(tensor([[0.1], [-0.1]]));
    const b = new Parameter(tensor([0.0]));
    const opt = new Adam([w, b], { lr: 0.01 });

    const x = tensor([[1, 0], [0, 1], [1, 1], [0, 0]]);
    const y = tensor([[1], [0], [1], [0]]);

    let firstLoss = null;
    let lastLoss = null;
    for (let epoch = 0; epoch < 50; epoch++) {
      const pred = add(matmul(x, w), b);
      const diff = sub(pred, y);
      const loss = mean(mul(diff, diff));
      if (epoch === 0) firstLoss = loss.item();
      lastLoss = loss.item();
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }
    expect(lastLoss).toBeLessThan(firstLoss);
  });
});
