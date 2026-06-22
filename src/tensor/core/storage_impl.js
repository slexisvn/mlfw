import { typedArrayCtor, dtypeSize } from '../types/dtype.js';
import { DeviceType } from '../types/device.js';

class CPUAllocator {
  allocate(count, dtype) {
    const Ctor = typedArrayCtor(dtype);
    return new Ctor(count);
  }

  free() {}
}

class MetaAllocator {
  allocate() {
    return null;
  }

  free() {}
}

const _ALLOCATORS = new Map([
  [DeviceType.CPU, new CPUAllocator()],
  [DeviceType.WASM, new CPUAllocator()],
  [DeviceType.META, new MetaAllocator()],
  [DeviceType.LAZY, new MetaAllocator()],
]);

export function getAllocator(deviceType) {
  return _ALLOCATORS.get(deviceType) || _ALLOCATORS.get(DeviceType.CPU);
}

export class StorageImpl {
  static #hostReadHook = null;
  static setHostReadHook(fn) { StorageImpl.#hostReadHook = fn; }

  constructor(data, nbytes, device, allocator) {
    this._data = data;
    this._nbytes = nbytes;
    this._device = device;
    this._allocator = allocator;
    this._refCount = 1;
  }

  static allocate(nbytes, dtype, device) {
    const allocator = getAllocator(device.type);
    const bytesPerElement = dtypeSize(dtype);
    const count = Math.max(Math.ceil(nbytes / bytesPerElement), 1);
    const data = allocator.allocate(count, dtype);
    return new StorageImpl(data, nbytes, device, allocator);
  }

  static fromData(data, device) {
    const nbytes = data ? data.byteLength : 0;
    const allocator = getAllocator(device.type);
    return new StorageImpl(data, nbytes, device, allocator);
  }

  retain() {
    this._refCount++;
    return this;
  }

  release() {
    this._refCount--;
    if (this._refCount === 0) {
      if (this._allocator && this._data) {
        this._allocator.free(this._data);
      }
      this._data = null;
    }
  }

  get refCount() {
    return this._refCount;
  }

  get data() {
    if (StorageImpl.#hostReadHook && this._data) StorageImpl.#hostReadHook(this._data);
    return this._data;
  }

  get rawData() {
    return this._data;
  }

  get nbytes() {
    return this._nbytes;
  }

  get device() {
    return this._device;
  }

  get isValid() {
    return this._refCount > 0 && this._data !== null;
  }

  get isMeta() {
    return this._data === null;
  }

  resize(newNbytes, dtype) {
    if (newNbytes <= this._nbytes && this._data) return;
    const bytesPerElement = dtypeSize(dtype);
    const count = Math.max(Math.ceil(newNbytes / bytesPerElement), 1);
    const newData = this._allocator.allocate(count, dtype);
    if (this._data && newData) {
      const copyLen = Math.min(this._data.length, newData.length);
      for (let i = 0; i < copyLen; i++) newData[i] = this._data[i];
    }
    if (this._allocator && this._data) {
      this._allocator.free(this._data);
    }
    this._data = newData;
    this._nbytes = newNbytes;
  }

  clone() {
    const allocator = getAllocator(this._device.type);
    let newData = null;
    if (this._data) {
      if (StorageImpl.#hostReadHook) StorageImpl.#hostReadHook(this._data);
      const Ctor = this._data.constructor;
      newData = new Ctor(this._data.length);
      newData.set(this._data);
    }
    return new StorageImpl(newData, this._nbytes, this._device, allocator);
  }
}
