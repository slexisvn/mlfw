import { describe, it, expect } from 'vitest';
import { tensor, Parameter, matmul, add, sub, mean, mul } from '../../src/index.js';
import { SGD } from '../../src/optim/sgd.js';

describe('SGD basic gradient descent', () => {
  it('updates w = w - lr * grad exactly', () => {
    const p = new Parameter(tensor([2.0, 4.0, 6.0]));
    p.grad = tensor([1.0, 2.0, 3.0]);
    const opt = new SGD([p], { lr: 0.1 });
    opt.step();
    const w = [...p._impl.storage.data];
    expect(w[0]).toBeCloseTo(1.9);
    expect(w[1]).toBeCloseTo(3.8);
    expect(w[2]).toBeCloseTo(5.7);
  });

  it('skips parameters with null grad', () => {
    const p = new Parameter(tensor([1.0, 2.0]));
    const opt = new SGD([p], { lr: 0.1 });
    expect(() => opt.step()).not.toThrow();
    expect([...p._impl.storage.data]).toEqual([1.0, 2.0]);
  });
});

describe('SGD with weight decay', () => {
  it('adds L2 penalty to gradient', () => {
    const p = new Parameter(tensor([10.0]));
    p.grad = tensor([1.0]);
    const opt = new SGD([p], { lr: 0.1, weightDecay: 0.5 });
    opt.step();
    expect(p._impl.storage.data[0]).toBeCloseTo(10.0 - 0.1 * (1.0 + 0.5 * 10.0));
  });
});

describe('SGD with momentum', () => {
  it('accumulates momentum buffer', () => {
    const p = new Parameter(tensor([0.0]));
    p.grad = tensor([10.0]);
    const opt = new SGD([p], { lr: 0.1, momentum: 0.9 });

    opt.step();
    expect(p._impl.storage.data[0]).toBeCloseTo(-1.0);

    p.grad = tensor([10.0]);
    opt.step();
    expect(p._impl.storage.data[0]).toBeCloseTo(-1.0 - 0.1 * (0.9 * 10.0 + 10.0));
  });

  it('applies dampening', () => {
    const p = new Parameter(tensor([0.0]));
    p.grad = tensor([10.0]);
    const opt = new SGD([p], { lr: 0.1, momentum: 0.9, dampening: 0.5 });

    opt.step();
    p.grad = tensor([10.0]);
    opt.step();

    const buf = 0.9 * 10.0 + 0.5 * 10.0;
    expect(p._impl.storage.data[0]).toBeCloseTo(-1.0 - 0.1 * buf);
  });
});

describe('SGD with nesterov', () => {
  it('nesterov differs from standard momentum', () => {
    const p1 = new Parameter(tensor([0.0]));
    p1.grad = tensor([10.0]);
    const standard = new SGD([p1], { lr: 0.1, momentum: 0.9 });
    standard.step();
    p1.grad = tensor([5.0]);
    standard.step();

    const p2 = new Parameter(tensor([0.0]));
    p2.grad = tensor([10.0]);
    const nesterov = new SGD([p2], { lr: 0.1, momentum: 0.9, nesterov: true });
    nesterov.step();
    p2.grad = tensor([5.0]);
    nesterov.step();

    expect(p1._impl.storage.data[0]).not.toBeCloseTo(p2._impl.storage.data[0]);
  });

  it('throws when momentum is 0 with nesterov', () => {
    const p = new Parameter(tensor([1.0]));
    expect(() => new SGD([p], { lr: 0.1, nesterov: true })).toThrow();
  });
});

describe('SGD with multiple parameter groups', () => {
  it('applies different learning rates per group', () => {
    const p1 = new Parameter(tensor([10.0]));
    const p2 = new Parameter(tensor([10.0]));
    p1.grad = tensor([1.0]);
    p2.grad = tensor([1.0]);
    const opt = new SGD([
      { params: [p1], lr: 0.1 },
      { params: [p2], lr: 1.0 },
    ]);
    opt.step();
    expect(p1._impl.storage.data[0]).toBeCloseTo(9.9);
    expect(p2._impl.storage.data[0]).toBeCloseTo(9.0);
  });
});

describe('SGD convergence', () => {
  it('decreases loss on a simple regression', () => {
    const w = new Parameter(tensor([[0.1], [-0.1]]));
    const b = new Parameter(tensor([0.0]));
    const opt = new SGD([w, b], { lr: 0.01 });

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
