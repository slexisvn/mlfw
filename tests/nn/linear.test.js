import { describe, it, expect } from 'vitest';
import { tensor, zeros, Linear } from '../../src/index.js';
import { constant_ } from '../../src/nn/init.js';

describe('Linear forward computes matmul(input, W^T) + bias', () => {
  it('produces correct output with known weights', () => {
    const layer = new Linear(3, 2);
    constant_(layer.weight, 0);
    layer.weight.data[0] = 1;
    layer.weight.data[1] = 0;
    layer.weight.data[2] = 0;
    layer.weight.data[3] = 0;
    layer.weight.data[4] = 0;
    layer.weight.data[5] = 1;
    constant_(layer.bias, 0);

    const input = tensor([[1, 2, 3]]);
    const out = layer.forward(input);
    const result = out.toArray();
    expect(result[0][0]).toBeCloseTo(1);
    expect(result[0][1]).toBeCloseTo(3);
  });

  it('adds bias correctly', () => {
    const layer = new Linear(2, 2);
    constant_(layer.weight, 1);
    layer.bias.data[0] = 10;
    layer.bias.data[1] = 20;

    const input = tensor([[1, 1]]);
    const out = layer.forward(input);
    const result = out.toArray();
    expect(result[0][0]).toBeCloseTo(12);
    expect(result[0][1]).toBeCloseTo(22);
  });

  it('works without bias', () => {
    const layer = new Linear(2, 3, false);
    constant_(layer.weight, 2);
    const input = tensor([[1, 1]]);
    const out = layer.forward(input);
    const result = [...out.data];
    expect(result.every(v => Math.abs(v - 4) < 1e-5)).toBe(true);
  });

  it('zero weight produces zero output regardless of input', () => {
    const layer = new Linear(3, 2);
    constant_(layer.weight, 0);
    constant_(layer.bias, 0);

    const input = tensor([[100, 200, 300]]);
    const out = layer.forward(input);
    expect(out.data[0]).toBeCloseTo(0);
  });
});
