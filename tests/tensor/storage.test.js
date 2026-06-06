import { describe, it, expect } from 'vitest';
import { Storage } from '../../src/tensor/core/storage.js';
import { ScalarType } from '../../src/tensor/types/dtype.js';
import { CPU_DEVICE } from '../../src/tensor/types/device.js';

describe('allocate', () => {
  it('allocates correct number of F32 elements from byte count', () => {
    const s = Storage.allocate(16, ScalarType.F32, CPU_DEVICE);
    expect(s.data.length).toBe(4);
  });

  it('allocates correct number of F64 elements from byte count', () => {
    const s = Storage.allocate(16, ScalarType.F64, CPU_DEVICE);
    expect(s.data.length).toBe(2);
  });
});

describe('clone', () => {
  it('creates independent copy that does not affect original', () => {
    const s = Storage.fromData(new Float32Array([10, 20, 30]), CPU_DEVICE);
    const c = s.clone();
    c.data[0] = 999;
    expect(s.data[0]).toBe(10);
    expect([...c.data]).toEqual([999, 20, 30]);
  });
});
