import { describe, it, expect } from 'vitest';
import { tensor, Parameter, matmul, add, sub, mean, mul } from '../../src/index.js';
import { Adam } from '../../src/optim/adam.js';
import { AdamW } from '../../src/optim/adamw.js';

describe('AdamW decoupled weight decay', () => {
  it('shrinks weight by (1 - lr * wd) when grad is zero', () => {
    const p = new Parameter(tensor([10.0]));
    p.grad = tensor([0.0]);
    const lr = 0.01;
    const wd = 0.1;
    const opt = new AdamW([p], { lr, weightDecay: wd });
    opt.step();
    const expected = 10.0 * (1 - lr * wd);
    expect(p._impl.storage.data[0]).toBeCloseTo(expected, 5);
  });

  it('differs from Adam when weightDecay > 0', () => {
    const p1 = new Parameter(tensor([5.0, -3.0]));
    const p2 = new Parameter(tensor([5.0, -3.0]));
    p1.grad = tensor([1.0, -1.0]);
    p2.grad = tensor([1.0, -1.0]);

    const adam = new Adam([p1], { lr: 0.01, weightDecay: 0.1 });
    const adamw = new AdamW([p2], { lr: 0.01, weightDecay: 0.1 });
    adam.step();
    adamw.step();

    expect(p1._impl.storage.data[0]).not.toBeCloseTo(p2._impl.storage.data[0], 5);
  });
});

describe('AdamW AMSGrad', () => {
  it('works with amsgrad enabled', () => {
    const p = new Parameter(tensor([1.0, 2.0]));
    const opt = new AdamW([p], { lr: 0.001, amsgrad: true });

    for (let i = 0; i < 5; i++) {
      p.grad = tensor([0.5 * (i + 1), -0.3 * (i + 1)]);
      opt.step();
    }
    const state = opt._getState(p);
    expect(state.maxExpAvgSq[0]).toBeGreaterThan(0);
    expect(state.maxExpAvgSq[1]).toBeGreaterThan(0);
  });
});

describe('AdamW convergence', () => {
  it('decreases loss on regression', () => {
    const w = new Parameter(tensor([[0.1], [-0.1]]));
    const b = new Parameter(tensor([0.0]));
    const opt = new AdamW([w, b], { lr: 0.01, weightDecay: 0.01 });

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

describe('AdamW stateDict round-trip', () => {
  it('preserves state across save/load', () => {
    const p = new Parameter(tensor([2.0, 4.0]));
    const opt = new AdamW([p], { lr: 0.005, weightDecay: 0.05 });

    for (let i = 0; i < 3; i++) {
      p.grad = tensor([1.0, -1.0]);
      opt.step();
    }
    const dict = opt.stateDict();
    const state = dict.state.get(0);
    expect(state.step).toBe(3);

    const p2 = new Parameter(tensor([2.0, 4.0]));
    const opt2 = new AdamW([p2], { lr: 0.1 });
    opt2.loadStateDict(dict);
    expect(opt2.paramGroups[0].lr).toBe(0.005);
    expect(opt2.paramGroups[0].weightDecay).toBe(0.05);
  });
});
