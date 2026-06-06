import { describe, it, expect } from 'vitest';
import { zeros } from '../../src/index.js';
import { Flatten } from '../../src/nn/modules/flatten.js';

describe('Flatten forward', () => {
  it('flattens [2, 3, 4, 5] with default startDim=1 to [2, 60]', () => {
    const input = zeros([2, 3, 4, 5]);
    const flatten = new Flatten();
    const out = flatten.forward(input);
    expect(out.shape).toEqual([2, 60]);
  });

  it('flattens [2, 3, 4] with startDim=0, endDim=-1 to [24]', () => {
    const input = zeros([2, 3, 4]);
    const flatten = new Flatten(0, -1);
    const out = flatten.forward(input);
    expect(out.shape).toEqual([24]);
  });

  it('flattens only middle dims with startDim=1, endDim=2', () => {
    const input = zeros([2, 3, 4, 5]);
    const flatten = new Flatten(1, 2);
    const out = flatten.forward(input);
    expect(out.shape).toEqual([2, 12, 5]);
  });

  it('negative startDim works', () => {
    const input = zeros([2, 3, 4]);
    const flatten = new Flatten(-2, -1);
    const out = flatten.forward(input);
    expect(out.shape).toEqual([2, 12]);
  });

  it('preserves data values', () => {
    const { tensor } = require('../../src/index.js');
    const input = tensor([[1, 2, 3], [4, 5, 6]]);
    const flatten = new Flatten(0, -1);
    const out = flatten.forward(input);
    expect([...out.data]).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('single batch element flattens correctly', () => {
    const input = zeros([1, 3, 4]);
    const flatten = new Flatten();
    const out = flatten.forward(input);
    expect(out.shape).toEqual([1, 12]);
  });
});
