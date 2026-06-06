import { describe, it, expect } from 'vitest';
import {
  zeros, ones, full, randn, arange, eye, linspace,
} from '../../src/tensor/factory/creation_ops.js';

describe('zeros', () => {
  it('fills all elements with 0', () => {
    const t = zeros([2, 3]);
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBe(0);
    }
  });
});

describe('ones', () => {
  it('fills all elements with 1', () => {
    const t = ones([3]);
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBe(1);
    }
  });
});

describe('full', () => {
  it('fills all elements with the given value', () => {
    const t = full([2, 2], 7.5);
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBeCloseTo(7.5);
    }
  });

  it('works with negative values', () => {
    const t = full([3], -2);
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBe(-2);
    }
  });
});

describe('arange', () => {
  it('generates [0,1,2,3,4] for arange(5)', () => {
    expect([...arange(5).data]).toEqual([0, 1, 2, 3, 4]);
  });

  it('generates [2,3,4] for arange(2,5)', () => {
    expect([...arange(2, 5).data]).toEqual([2, 3, 4]);
  });

  it('generates [0, 0.5] for arange(0, 1, 0.5)', () => {
    const t = arange(0, 1, 0.5);
    expect(t.data[0]).toBeCloseTo(0);
    expect(t.data[1]).toBeCloseTo(0.5);
  });

  it('generates empty tensor for arange(5, 0)', () => {
    expect(arange(5, 0).numel).toBe(0);
  });
});

describe('eye', () => {
  it('creates 3x3 identity matrix', () => {
    expect(eye(3).toArray()).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
  });

  it('creates rectangular identity 2x3', () => {
    expect(eye(2, 3).toArray()).toEqual([
      [1, 0, 0],
      [0, 1, 0],
    ]);
  });
});

describe('linspace', () => {
  it('generates evenly spaced values with correct endpoints', () => {
    const t = linspace(0, 1, 5);
    expect(t.data[0]).toBeCloseTo(0);
    expect(t.data[1]).toBeCloseTo(0.25);
    expect(t.data[2]).toBeCloseTo(0.5);
    expect(t.data[3]).toBeCloseTo(0.75);
    expect(t.data[4]).toBeCloseTo(1.0);
  });

  it('returns single start value for steps=1', () => {
    const t = linspace(0, 1, 1);
    expect(t.data[0]).toBeCloseTo(0);
    expect(t.data.length).toBe(1);
  });

  it('handles negative range', () => {
    const t = linspace(1, -1, 3);
    expect(t.data[0]).toBeCloseTo(1);
    expect(t.data[1]).toBeCloseTo(0);
    expect(t.data[2]).toBeCloseTo(-1);
  });
});

describe('randn', () => {
  it('generates non-zero values', () => {
    const t = randn([100]);
    let allZero = true;
    for (let i = 0; i < t.data.length; i++) {
      if (t.data[i] !== 0) { allZero = false; break; }
    }
    expect(allZero).toBe(false);
  });

  it('has mean approximately 0', () => {
    const t = randn([1000]);
    let sum = 0;
    for (let i = 0; i < t.data.length; i++) sum += t.data[i];
    expect(Math.abs(sum / t.data.length)).toBeLessThan(0.2);
  });
});
