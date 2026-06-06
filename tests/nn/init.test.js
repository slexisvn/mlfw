import { describe, it, expect } from 'vitest';
import { tensor, zeros, ones } from '../../src/index.js';
import {
  uniform_, normal_, zeros_, ones_, constant_,
  xavier_uniform_, xavier_normal_,
  kaiming_uniform_, kaiming_normal_,
  _calculateFanInFanOut,
} from '../../src/nn/init.js';

describe('uniform_', () => {
  it('fills values within [a, b] range', () => {
    const t = zeros([200]);
    uniform_(t, -2, 2);
    const data = [...t.data];
    expect(data.every(v => v >= -2 && v <= 2)).toBe(true);
    expect(data.some(v => v !== 0)).toBe(true);
  });

  it('respects custom range bounds', () => {
    const t = zeros([500]);
    uniform_(t, 10, 11);
    const data = [...t.data];
    expect(data.every(v => v >= 10 && v <= 11)).toBe(true);
  });
});

describe('normal_', () => {
  it('produces values with approximate mean and std', () => {
    const t = zeros([2000]);
    normal_(t, 5, 2);
    const data = [...t.data];
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
    expect(mean).toBeCloseTo(5, 0);
    expect(Math.sqrt(variance)).toBeCloseTo(2, 0);
  });

  it('handles odd-length tensors (Box-Muller pairs)', () => {
    const t = zeros([7]);
    normal_(t, 0, 1);
    const data = [...t.data];
    expect(data.length).toBe(7);
    expect(data.every(v => isFinite(v))).toBe(true);
    expect(data.some(v => v !== 0)).toBe(true);
  });
});

describe('zeros_', () => {
  it('fills all elements with zero', () => {
    const t = ones([3, 4]);
    zeros_(t);
    expect([...t.data].every(v => v === 0)).toBe(true);
  });
});

describe('ones_', () => {
  it('fills all elements with one', () => {
    const t = zeros([3, 4]);
    ones_(t);
    expect([...t.data].every(v => v === 1)).toBe(true);
  });
});

describe('constant_', () => {
  it('fills all elements with specified value', () => {
    const t = zeros([2, 3]);
    constant_(t, 42);
    expect([...t.data].every(v => v === 42)).toBe(true);
  });
});

describe('_calculateFanInFanOut', () => {
  it('computes fan_in and fan_out for 2D weight', () => {
    const t = zeros([5, 3]);
    const { fanIn, fanOut } = _calculateFanInFanOut(t);
    expect(fanIn).toBe(3);
    expect(fanOut).toBe(5);
  });

  it('includes receptive field for conv weights [out, in, kH, kW]', () => {
    const t = zeros([16, 3, 5, 5]);
    const { fanIn, fanOut } = _calculateFanInFanOut(t);
    expect(fanIn).toBe(3 * 5 * 5);
    expect(fanOut).toBe(16 * 5 * 5);
  });

  it('throws for 1D tensor', () => {
    const t = zeros([5]);
    expect(() => _calculateFanInFanOut(t)).toThrow(/at least 2D/);
  });
});

describe('xavier_uniform_', () => {
  it('fills within correct bound: sqrt(6 / (fan_in + fan_out))', () => {
    const t = zeros([10, 20]);
    xavier_uniform_(t);
    const bound = Math.sqrt(6.0 / (20 + 10));
    const data = [...t.data];
    expect(data.every(v => v >= -bound - 1e-6 && v <= bound + 1e-6)).toBe(true);
    expect(data.some(v => v !== 0)).toBe(true);
  });

  it('gain scales the bound', () => {
    const t = zeros([10, 20]);
    const gain = 2.0;
    xavier_uniform_(t, gain);
    const std = gain * Math.sqrt(2.0 / (20 + 10));
    const bound = Math.sqrt(3.0) * std;
    const data = [...t.data];
    expect(data.every(v => v >= -bound - 1e-6 && v <= bound + 1e-6)).toBe(true);
  });
});

describe('xavier_normal_', () => {
  it('produces values with std ~ sqrt(2 / (fan_in + fan_out))', () => {
    const t = zeros([100, 100]);
    xavier_normal_(t);
    const expectedStd = Math.sqrt(2.0 / (100 + 100));
    const data = [...t.data];
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
    expect(Math.sqrt(variance)).toBeCloseTo(expectedStd, 1);
  });
});

describe('kaiming_uniform_', () => {
  it('fills within correct bound for fan_in mode with relu', () => {
    const t = zeros([10, 20]);
    kaiming_uniform_(t, 0, 'fan_in', 'relu');
    const gain = Math.sqrt(2.0);
    const std = gain / Math.sqrt(20);
    const bound = Math.sqrt(3.0) * std;
    const data = [...t.data];
    expect(data.every(v => v >= -bound - 1e-6 && v <= bound + 1e-6)).toBe(true);
  });

  it('uses fan_out when specified', () => {
    const t = zeros([10, 20]);
    kaiming_uniform_(t, 0, 'fan_out', 'relu');
    const gain = Math.sqrt(2.0);
    const std = gain / Math.sqrt(10);
    const bound = Math.sqrt(3.0) * std;
    const data = [...t.data];
    expect(data.every(v => v >= -bound - 1e-6 && v <= bound + 1e-6)).toBe(true);
  });
});

describe('kaiming_normal_', () => {
  it('produces values with std ~ gain / sqrt(fan)', () => {
    const t = zeros([100, 100]);
    kaiming_normal_(t, 0, 'fan_in', 'relu');
    const expectedStd = Math.sqrt(2.0) / Math.sqrt(100);
    const data = [...t.data];
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
    expect(Math.sqrt(variance)).toBeCloseTo(expectedStd, 1);
  });
});
