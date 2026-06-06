import { describe, it, expect } from 'vitest';
import { tensor } from '../../src/index.js';
import { leaky_relu, softmax, log_softmax } from '../../src/nn/functional/activation.js';
import { ReLU, LeakyReLU, Softmax } from '../../src/nn/modules/activation.js';

describe('leaky_relu', () => {
  it('passes positive values unchanged and scales negatives', () => {
    const input = tensor([-2, -1, 0, 1, 2]);
    const out = leaky_relu(input, 0.1);
    const result = [...out.data];
    expect(result[0]).toBeCloseTo(-0.2);
    expect(result[1]).toBeCloseTo(-0.1);
    expect(result[2]).toBeCloseTo(0);
    expect(result[3]).toBeCloseTo(1);
    expect(result[4]).toBeCloseTo(2);
  });

  it('uses custom negative slope', () => {
    const input = tensor([-4]);
    const out = leaky_relu(input, 0.5);
    expect(out.data[0]).toBeCloseTo(-2);
  });
});

describe('softmax', () => {
  it('outputs sum to 1 along last dim', () => {
    const input = tensor([[1, 2, 3], [1, 1, 1]]);
    const out = softmax(input, -1);
    const data = [...out.data];
    const row0Sum = data[0] + data[1] + data[2];
    const row1Sum = data[3] + data[4] + data[5];
    expect(row0Sum).toBeCloseTo(1);
    expect(row1Sum).toBeCloseTo(1);
  });

  it('highest input gets highest probability', () => {
    const input = tensor([1, 5, 2]);
    const out = softmax(input, -1);
    const data = [...out.data];
    expect(data[1]).toBeGreaterThan(data[0]);
    expect(data[1]).toBeGreaterThan(data[2]);
  });

  it('equal inputs produce equal probabilities', () => {
    const input = tensor([3, 3, 3]);
    const out = softmax(input, -1);
    const data = [...out.data];
    expect(data[0]).toBeCloseTo(1 / 3, 4);
    expect(data[1]).toBeCloseTo(1 / 3, 4);
    expect(data[2]).toBeCloseTo(1 / 3, 4);
  });
});

describe('log_softmax', () => {
  it('outputs sum to 0 in exp space', () => {
    const input = tensor([1, 2, 3]);
    const out = log_softmax(input, -1);
    const data = [...out.data];
    const expSum = Math.exp(data[0]) + Math.exp(data[1]) + Math.exp(data[2]);
    expect(expSum).toBeCloseTo(1);
  });

  it('all values are negative or zero', () => {
    const input = tensor([1, 2, 3]);
    const out = log_softmax(input, -1);
    const data = [...out.data];
    expect(data.every(v => v <= 0 + 1e-6)).toBe(true);
  });
});

describe('ReLU module forward', () => {
  it('zeros negatives and passes positives', () => {
    const m = new ReLU();
    const input = tensor([-3, -1, 0, 1, 5]);
    const out = m.forward(input);
    const result = [...out.data];
    expect(result[0]).toBe(0);
    expect(result[1]).toBe(0);
    expect(result[2]).toBe(0);
    expect(result[3]).toBe(1);
    expect(result[4]).toBe(5);
  });
});

describe('LeakyReLU module forward', () => {
  it('applies negative slope from constructor', () => {
    const m = new LeakyReLU(0.2);
    const input = tensor([-5, 3]);
    const out = m.forward(input);
    expect(out.data[0]).toBeCloseTo(-1);
    expect(out.data[1]).toBeCloseTo(3);
  });
});

describe('Softmax module forward', () => {
  it('applies softmax along specified dim', () => {
    const m = new Softmax(-1);
    const input = tensor([1, 1, 1]);
    const out = m.forward(input);
    const data = [...out.data];
    expect(data.every(v => Math.abs(v - 1 / 3) < 1e-4)).toBe(true);
  });
});
