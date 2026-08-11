import { describe, it, expect } from 'vitest';
import { tensor, Parameter } from '../../src/index.js';
import { SGD } from '../../src/optim/sgd.js';
import { GradScaler } from '../../src/optim/grad_scaler.js';

function paramWithGrad(value, grad) {
  const p = new Parameter(tensor(value));
  p.grad = tensor(value.map(() => 0));
  p.grad._impl.storage.data.set(grad);
  return p;
}

describe('GradScaler loss scaling', () => {
  it('scale() multiplies the loss by the current scale', () => {
    const scaler = new GradScaler({ initScale: 1024 });
    const loss = tensor([2.0, -1.5]);
    scaler.scale(loss);
    expect(Array.from(loss._impl.storage.data)).toEqual([2048, -1536]);
  });

  it('unscale_() divides grads back by the scale', () => {
    const scaler = new GradScaler({ initScale: 8 });
    const p = paramWithGrad([1, 2], [16, 24]);
    const opt = new SGD([p], { lr: 0.1 });
    const foundInf = scaler.unscale_(opt);
    expect(foundInf).toBe(false);
    expect(Array.from(p.grad._impl.storage.data)).toEqual([2, 3]);
  });

  it('disabled scaler is a pass-through (scale=1, scale() no-op)', () => {
    const scaler = new GradScaler({ enabled: false, initScale: 1024 });
    expect(scaler.getScale()).toBe(1.0);
    const loss = tensor([3.0]);
    scaler.scale(loss);
    expect(loss._impl.storage.data[0]).toBe(3.0);
  });
});

describe('GradScaler overflow handling skips the step and backs off', () => {
  it('a non-finite grad makes step() skip the update and update() halve the scale', () => {
    const scaler = new GradScaler({ initScale: 1024, backoffFactor: 0.5 });
    const p = paramWithGrad([5.0], [1.0]);
    p.grad._impl.storage.data[0] = Infinity;
    const opt = new SGD([p], { lr: 0.1 });

    const stepped = scaler.step(opt);
    expect(stepped).toBe(false);
    expect(p._impl.storage.data[0]).toBe(5.0);

    scaler.update();
    expect(scaler.getScale()).toBe(512);
    expect(scaler.growthTracker).toBe(0);
  });

  it('a clean step applies the update and is unscaled exactly once', () => {
    const scaler = new GradScaler({ initScale: 4 });
    const p = paramWithGrad([1.0], [8.0]);
    const opt = new SGD([p], { lr: 0.5 });

    const stepped = scaler.step(opt);
    expect(stepped).toBe(true);
    expect(p._impl.storage.data[0]).toBeCloseTo(1.0 - 0.5 * 2.0, 6);
  });
});

describe('GradScaler grows the scale after a clean growth interval', () => {
  it('doubles the scale after growthInterval consecutive non-overflow updates', () => {
    const scaler = new GradScaler({ initScale: 1000, growthInterval: 3, growthFactor: 2 });
    for (let i = 0; i < 2; i++) scaler.update();
    expect(scaler.getScale()).toBe(1000);
    scaler.update();
    expect(scaler.getScale()).toBe(2000);
    expect(scaler.growthTracker).toBe(0);
  });

  it('an overflow update resets the growth tracker', () => {
    const scaler = new GradScaler({ initScale: 1000, growthInterval: 3 });
    scaler.update();
    scaler.update();
    expect(scaler.growthTracker).toBe(2);
    scaler._foundInf = true;
    scaler.update();
    expect(scaler.growthTracker).toBe(0);
    expect(scaler.getScale()).toBe(500);
  });
});
