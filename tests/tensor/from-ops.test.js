import { describe, it, expect } from 'vitest';
import { tensor, fromBuffer, scalar } from '../../src/tensor/factory/from_ops.js';
import { ScalarType } from '../../src/tensor/types/dtype.js';

describe('tensor from nested array', () => {
  it('creates 2D tensor with correct shape and values', () => {
    const t = tensor([[1, 2], [3, 4]]);
    expect(t.shape).toEqual([2, 2]);
    expect(t.toArray()).toEqual([[1, 2], [3, 4]]);
  });

  it('creates 3D tensor', () => {
    const t = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    expect(t.shape).toEqual([2, 2, 2]);
    expect(t.toArray()).toEqual([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
  });

  it('creates 1D tensor from flat array', () => {
    const t = tensor([10, 20, 30]);
    expect(t.shape).toEqual([3]);
    expect([...t.data]).toEqual([10, 20, 30]);
  });
});

describe('tensor with explicit shape', () => {
  it('reshapes flat array to given shape', () => {
    const t = tensor([1, 2, 3, 4, 5, 6], { shape: [2, 3] });
    expect(t.shape).toEqual([2, 3]);
    expect(t.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
  });
});

describe('tensor from typed array', () => {
  it('copies data so source modification has no effect', () => {
    const src = new Float32Array([1, 2, 3]);
    const t = tensor(src);
    src[0] = 999;
    expect(t.data[0]).toBe(1);
  });
});

describe('tensor from number', () => {
  it('creates scalar tensor', () => {
    const t = tensor(5);
    expect(t.shape).toEqual([]);
    expect(t.item()).toBe(5);
  });
});

describe('tensor with dtype override', () => {
  it('uses specified dtype', () => {
    const t = tensor([1, 2, 3], { dtype: ScalarType.F64 });
    expect(t.data).toBeInstanceOf(Float64Array);
  });
});

describe('tensor unsupported input', () => {
  it('throws for string input', () => {
    expect(() => tensor('abc')).toThrow();
  });
});

describe('fromBuffer', () => {
  it('creates tensor from typed array with shape', () => {
    const buf = new Float32Array([1, 2, 3, 4]);
    const t = fromBuffer(buf, [2, 2], ScalarType.F32);
    expect(t.shape).toEqual([2, 2]);
    expect(t.toArray()).toEqual([[1, 2], [3, 4]]);
  });
});

describe('scalar', () => {
  it('creates 0D tensor', () => {
    const t = scalar(3.14);
    expect(t.shape).toEqual([]);
    expect(t.item()).toBeCloseTo(3.14);
  });

  it('supports dtype option', () => {
    const t = scalar(42, { dtype: ScalarType.I32 });
    expect(t.data).toBeInstanceOf(Int32Array);
    expect(t.item()).toBe(42);
  });
});
