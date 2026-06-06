import { describe, it, expect } from 'vitest';
import { tensor, ones } from '../../src/index.js';
import { dropout } from '../../src/nn/functional/dropout.js';
import { Dropout } from '../../src/nn/modules/dropout.js';

describe('dropout functional', () => {
  it('returns input unchanged when training=false', () => {
    const input = tensor([1, 2, 3, 4, 5]);
    const out = dropout(input, 0.5, false);
    expect([...out.data]).toEqual([1, 2, 3, 4, 5]);
  });

  it('returns input unchanged when p=0', () => {
    const input = tensor([1, 2, 3, 4, 5]);
    const out = dropout(input, 0, true);
    expect([...out.data]).toEqual([1, 2, 3, 4, 5]);
  });

  it('returns all zeros when p=1', () => {
    const input = tensor([1, 2, 3, 4, 5]);
    const out = dropout(input, 1, true);
    expect([...out.data].every(v => v === 0)).toBe(true);
  });

  it('scales surviving values by 1/(1-p)', () => {
    const input = ones([1000]);
    const out = dropout(input, 0.5, true);
    const data = [...out.data];
    const nonZero = data.filter(v => v !== 0);
    if (nonZero.length > 0) {
      expect(nonZero[0]).toBeCloseTo(2, 4);
    }
  });

  it('zeros roughly p fraction of elements', () => {
    const input = ones([5000]);
    const out = dropout(input, 0.3, true);
    const data = [...out.data];
    const zeroCount = data.filter(v => v === 0).length;
    const fraction = zeroCount / data.length;
    expect(fraction).toBeGreaterThan(0.15);
    expect(fraction).toBeLessThan(0.45);
  });
});

describe('Dropout module', () => {
  it('drops in training mode', () => {
    const m = new Dropout(0.99);
    const input = ones([100]);
    const out = m.forward(input);
    const data = [...out.data];
    const zeroCount = data.filter(v => v === 0).length;
    expect(zeroCount).toBeGreaterThan(80);
  });

  it('passes through in eval mode', () => {
    const m = new Dropout(0.5);
    m.eval();
    const input = tensor([1, 2, 3]);
    const out = m.forward(input);
    expect([...out.data]).toEqual([1, 2, 3]);
  });
});
