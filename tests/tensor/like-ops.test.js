import { describe, it, expect } from 'vitest';
import { ones } from '../../src/tensor/factory/creation_ops.js';
import {
  zerosLike, onesLike, fullLike, randnLike,
} from '../../src/tensor/factory/like_ops.js';

describe('zerosLike', () => {
  it('creates all-zero tensor matching source shape', () => {
    const t = zerosLike(ones([3, 4]));
    expect(t.shape).toEqual([3, 4]);
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBe(0);
    }
  });

  it('does not share storage with source', () => {
    const src = ones([3]);
    const t = zerosLike(src);
    t.data[0] = 99;
    expect(src.data[0]).toBe(1);
  });
});

describe('onesLike', () => {
  it('creates all-ones tensor matching source shape', () => {
    const t = onesLike(ones([2, 5]));
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBe(1);
    }
  });
});

describe('fullLike', () => {
  it('fills with given value matching source shape', () => {
    const t = fullLike(ones([4]), 42);
    for (let i = 0; i < t.data.length; i++) {
      expect(t.data[i]).toBe(42);
    }
  });
});

describe('randnLike', () => {
  it('produces non-constant values matching source shape', () => {
    const t = randnLike(ones([50]));
    const unique = new Set(t.data);
    expect(unique.size).toBeGreaterThan(1);
  });
});
