import { describe, it, expect } from 'vitest';
import { TensorImpl } from '../../src/tensor/core/tensor_impl.js';
import { Storage } from '../../src/tensor/core/storage.js';
import { ScalarType } from '../../src/tensor/types/dtype.js';
import { CPU_DEVICE } from '../../src/tensor/types/device.js';
import { DispatchKey } from '../../src/dispatcher/dispatch_key.js';
import { MemoryFormat } from '../../src/tensor/types/layout.js';

function makeImpl(sizes, strides) {
  const storage = Storage.allocate(256, ScalarType.F32, CPU_DEVICE);
  return new TensorImpl(storage, 0, sizes, strides, ScalarType.F32, CPU_DEVICE);
}

describe('TensorImpl.size and stride with negative index', () => {
  it('size(-1) returns last dim', () => {
    const impl = makeImpl([2, 3, 4], [12, 4, 1]);
    expect(impl.size(-1)).toBe(4);
    expect(impl.size(-2)).toBe(3);
    expect(impl.size(-3)).toBe(2);
  });

  it('stride(-1) returns last stride', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(impl.stride(-1)).toBe(1);
    expect(impl.stride(-2)).toBe(3);
  });
});

describe('TensorImpl.isContiguous caching', () => {
  it('contiguous tensor caches true', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(impl.isContiguous()).toBe(true);
    expect(impl.isContiguous()).toBe(true);
  });

  it('non-contiguous tensor caches false', () => {
    const impl = makeImpl([2, 3], [1, 2]);
    expect(impl.isContiguous()).toBe(false);
    expect(impl.isContiguous()).toBe(false);
  });

  it('setSizesAndStrides invalidates contiguous cache', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(impl.isContiguous()).toBe(true);
    impl.setSizesAndStrides([3, 2], [1, 3]);
    expect(impl.isContiguous()).toBe(false);
  });

  it('setSizesAndStrides updates numel', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(impl.numel()).toBe(6);
    impl.setSizesAndStrides([4, 5], null);
    expect(impl.numel()).toBe(20);
  });
});

describe('TensorImpl.isContiguous with MemoryFormat', () => {
  it('unknown format returns false', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(impl.isContiguous('unknown_format')).toBe(false);
  });
});

describe('TensorImpl dispatch key management', () => {
  it('initial keySet has CPU key', () => {
    const impl = makeImpl([2], [1]);
    expect(impl.keySet().has(DispatchKey.CPU)).toBe(true);
  });

  it('setAutogradMeta with requiresGrad adds autograd key', () => {
    const impl = makeImpl([2], [1]);
    impl.setAutogradMeta({ requiresGrad: true });
    expect(impl.keySet().has(DispatchKey.AUTOGRAD_CPU)).toBe(true);
  });

  it('setAutogradMeta without requiresGrad keeps only backend key', () => {
    const impl = makeImpl([2], [1]);
    impl.setAutogradMeta({ requiresGrad: false });
    expect(impl.keySet().has(DispatchKey.AUTOGRAD_CPU)).toBe(false);
    expect(impl.keySet().has(DispatchKey.CPU)).toBe(true);
  });
});

describe('TensorImpl version tracking', () => {
  it('starts at 0 and increments', () => {
    const impl = makeImpl([2], [1]);
    expect(impl.version).toBe(0);
    impl.bumpVersion();
    expect(impl.version).toBe(1);
    impl.bumpVersion();
    expect(impl.version).toBe(2);
  });
});

describe('TensorImpl.shallowCopyFrom', () => {
  it('copies all metadata and invalidates contiguous cache', () => {
    const a = makeImpl([2, 3], [3, 1]);
    const b = makeImpl([4, 5], [1, 4]);
    a.isContiguous();

    a.shallowCopyFrom(b);
    expect(a.sizes()).toEqual([4, 5]);
    expect(a.strides()).toEqual([1, 4]);
    expect(a.numel()).toBe(20);
    expect(a.storage).toBe(b.storage);
    expect(a.isContiguous()).toBe(false);
  });
});

describe('TensorImpl sizes/strides are frozen', () => {
  it('sizes() returns frozen array', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(Object.isFrozen(impl.sizes())).toBe(true);
  });

  it('strides() returns frozen array', () => {
    const impl = makeImpl([2, 3], [3, 1]);
    expect(Object.isFrozen(impl.strides())).toBe(true);
  });
});
