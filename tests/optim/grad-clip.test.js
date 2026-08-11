import { describe, it, expect } from 'vitest';
import { tensor, Parameter } from '../../src/index.js';
import { clipGradNorm_, clipGradValue_ } from '../../src/optim/utils.js';

describe('clipGradNorm_', () => {
  it('does not modify grads when norm is below maxNorm', () => {
    const p = new Parameter(tensor([1.0, 0.0]));
    p.grad = tensor([0.3, 0.4]);
    const norm = clipGradNorm_([p], 10.0);
    expect(norm).toBeCloseTo(0.5);
    expect(p.grad._impl.storage.data[0]).toBeCloseTo(0.3);
    expect(p.grad._impl.storage.data[1]).toBeCloseTo(0.4);
  });

  it('scales grads when norm exceeds maxNorm', () => {
    const p1 = new Parameter(tensor([3.0]));
    const p2 = new Parameter(tensor([4.0]));
    p1.grad = tensor([3.0]);
    p2.grad = tensor([4.0]);
    const norm = clipGradNorm_([p1, p2], 1.0);
    expect(norm).toBeCloseTo(5.0);

    const g1 = p1.grad._impl.storage.data[0];
    const g2 = p2.grad._impl.storage.data[0];
    const newNorm = Math.sqrt(g1 * g1 + g2 * g2);
    expect(newNorm).toBeCloseTo(1.0, 4);
  });

  it('handles inf normType', () => {
    const p = new Parameter(tensor([1.0, 2.0]));
    p.grad = tensor([-5.0, 3.0]);
    const norm = clipGradNorm_([p], 2.0, Infinity);
    expect(norm).toBeCloseTo(5.0);

    const d = p.grad._impl.storage.data;
    const maxAbs = Math.max(Math.abs(d[0]), Math.abs(d[1]));
    expect(maxAbs).toBeLessThanOrEqual(2.0 + 1e-5);
  });

  it('returns total norm as a number', () => {
    const p = new Parameter(tensor([3.0, 4.0]));
    p.grad = tensor([3.0, 4.0]);
    const result = clipGradNorm_([p], 100.0);
    expect(typeof result).toBe('number');
    expect(result).toBeCloseTo(5.0);
  });

  it('handles parameters with null grad', () => {
    const p1 = new Parameter(tensor([1.0]));
    const p2 = new Parameter(tensor([2.0]));
    p1.grad = tensor([6.0]);
    const norm = clipGradNorm_([p1, p2], 1.0);
    expect(norm).toBeCloseTo(6.0);
  });

  it('accepts a generator', () => {
    const p = new Parameter(tensor([3.0, 4.0]));
    p.grad = tensor([3.0, 4.0]);
    function* gen() { yield p; }
    const norm = clipGradNorm_(gen(), 1.0);
    expect(norm).toBeCloseTo(5.0);
  });
});

describe('clipGradValue_', () => {
  it('clamps gradient values to [-clipValue, clipValue]', () => {
    const p = new Parameter(tensor([1.0, 2.0, 3.0]));
    p.grad = tensor([10.0, -20.0, 0.5]);
    clipGradValue_([p], 5.0);
    const d = p.grad._impl.storage.data;
    expect(d[0]).toBe(5.0);
    expect(d[1]).toBe(-5.0);
    expect(d[2]).toBeCloseTo(0.5);
  });

  it('does not modify values within range', () => {
    const p = new Parameter(tensor([1.0]));
    p.grad = tensor([2.5]);
    clipGradValue_([p], 5.0);
    expect(p.grad._impl.storage.data[0]).toBeCloseTo(2.5);
  });

  it('handles null grad gracefully', () => {
    const p = new Parameter(tensor([1.0]));
    expect(() => clipGradValue_([p], 1.0)).not.toThrow();
  });
});
