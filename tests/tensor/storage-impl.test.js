import { describe, it, expect } from 'vitest';
import { StorageImpl } from '../../src/tensor/core/storage_impl.js';
import { ScalarType } from '../../src/tensor/types/dtype.js';
import { CPU_DEVICE } from '../../src/tensor/types/device.js';

describe('retain and release lifecycle', () => {
  it('release to zero nullifies data so reads fail', () => {
    const s = StorageImpl.allocate(8, ScalarType.F32, CPU_DEVICE);
    s.data[0] = 42;
    s.release();
    expect(s.data).toBeNull();
  });

  it('retain keeps data alive through one extra release', () => {
    const s = StorageImpl.allocate(8, ScalarType.F32, CPU_DEVICE);
    s.data[0] = 99;
    s.retain();
    s.release();
    expect(s.data[0]).toBe(99);
    s.release();
    expect(s.data).toBeNull();
  });

  it('multiple retains need matching releases to free', () => {
    const s = StorageImpl.allocate(4, ScalarType.F32, CPU_DEVICE);
    s.retain();
    s.retain();
    s.release();
    s.release();
    expect(s.data).not.toBeNull();
    s.release();
    expect(s.data).toBeNull();
  });
});

describe('resize', () => {
  it('grows storage and preserves existing data', () => {
    const s = StorageImpl.allocate(8, ScalarType.F32, CPU_DEVICE);
    s.data[0] = 10;
    s.data[1] = 20;
    s.resize(16, ScalarType.F32);
    expect(s.data.length).toBe(4);
    expect(s.data[0]).toBe(10);
    expect(s.data[1]).toBe(20);
  });

  it('does not shrink when new size is smaller', () => {
    const s = StorageImpl.allocate(16, ScalarType.F32, CPU_DEVICE);
    const originalData = s.data;
    s.resize(8, ScalarType.F32);
    expect(s.data).toBe(originalData);
  });
});

describe('clone', () => {
  it('creates independent copy — modifying clone does not affect original', () => {
    const s = StorageImpl.allocate(12, ScalarType.F32, CPU_DEVICE);
    s.data[0] = 100;
    s.data[1] = 200;
    s.data[2] = 300;
    const c = s.clone();
    expect([...c.data]).toEqual([100, 200, 300]);
    c.data[0] = 999;
    expect(s.data[0]).toBe(100);
  });

  it('clone has independent lifecycle from original', () => {
    const s = StorageImpl.allocate(8, ScalarType.F32, CPU_DEVICE);
    s.data[0] = 42;
    const c = s.clone();
    s.release();
    expect(s.data).toBeNull();
    expect(c.data[0]).toBe(42);
  });
});

describe('allocate byte-to-element calculation', () => {
  it('24 bytes / 8 bytes-per-F64 = 3 elements', () => {
    const s = StorageImpl.allocate(24, ScalarType.F64, CPU_DEVICE);
    expect(s.data.length).toBe(3);
  });

  it('12 bytes / 4 bytes-per-I32 = 3 elements', () => {
    const s = StorageImpl.allocate(12, ScalarType.I32, CPU_DEVICE);
    expect(s.data.length).toBe(3);
  });

  it('allocates at least 1 element even for 0 bytes', () => {
    const s = StorageImpl.allocate(0, ScalarType.F32, CPU_DEVICE);
    expect(s.data.length).toBeGreaterThanOrEqual(1);
  });
});
