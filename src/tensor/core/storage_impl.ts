import { typedArrayCtor, dtypeSize } from '../types/dtype.js';
import type { DType, NumericTypedArray } from '../types/dtype.js';
import { DeviceType } from '../types/device.js';
import type { Device, DeviceTypeName } from '../types/device.js';

class CPUAllocator {
  allocate(count: number, dtype: DType): NumericTypedArray {
    const Ctor = typedArrayCtor(dtype);
    return new Ctor(count);
  }

  free(_data?: NumericTypedArray) {}
}

class MetaAllocator {
  allocate(): null {
    return null;
  }

  free(_data?: NumericTypedArray | null) {}
}

type Allocator = CPUAllocator | MetaAllocator;
type NumericSettable = { set(array: ArrayLike<number | bigint>, offset?: number): void };

const _ALLOCATORS = new Map<DeviceTypeName, Allocator>([
  [DeviceType.CPU, new CPUAllocator()],
  [DeviceType.WASM, new CPUAllocator()],
  [DeviceType.META, new MetaAllocator()],
  [DeviceType.LAZY, new MetaAllocator()],
]);

export function getAllocator(deviceType: DeviceTypeName): Allocator {
  return _ALLOCATORS.get(deviceType) || _ALLOCATORS.get(DeviceType.CPU)!;
}

export class StorageImpl {
  static #hostReadHook: ((data: NumericTypedArray) => void) | null = null;
  private _data: NumericTypedArray | null;
  private _nbytes: number;
  private readonly _device: Device;
  private readonly _allocator: Allocator;
  private _refCount: number;

  static setHostReadHook(fn: ((data: NumericTypedArray) => void) | null) { StorageImpl.#hostReadHook = fn; }

  constructor(data: NumericTypedArray | null, nbytes: number, device: Device, allocator: Allocator) {
    this._data = data;
    this._nbytes = nbytes;
    this._device = device;
    this._allocator = allocator;
    this._refCount = 1;
  }

  static allocate(nbytes: number, dtype: DType, device: Device): StorageImpl {
    const allocator = getAllocator(device.type);
    const bytesPerElement = dtypeSize(dtype);
    const count = Math.max(Math.ceil(nbytes / bytesPerElement), 1);
    const data = allocator.allocate(count, dtype);
    return new StorageImpl(data, nbytes, device, allocator);
  }

  static fromData(data: NumericTypedArray | null, device: Device): StorageImpl {
    const nbytes = data ? data.byteLength : 0;
    const allocator = getAllocator(device.type);
    return new StorageImpl(data, nbytes, device, allocator);
  }

  retain(): StorageImpl {
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

  get refCount(): number {
    return this._refCount;
  }

  get data(): NumericTypedArray | null {
    if (StorageImpl.#hostReadHook && this._data) StorageImpl.#hostReadHook(this._data);
    return this._data;
  }

  get rawData(): NumericTypedArray | null {
    return this._data;
  }

  get nbytes(): number {
    return this._nbytes;
  }

  get device(): Device {
    return this._device;
  }

  get isValid(): boolean {
    return this._refCount > 0 && this._data !== null;
  }

  get isMeta(): boolean {
    return this._data === null;
  }

  resize(newNbytes: number, dtype: DType) {
    if (newNbytes <= this._nbytes && this._data) return;
    const bytesPerElement = dtypeSize(dtype);
    const count = Math.max(Math.ceil(newNbytes / bytesPerElement), 1);
    const newData = this._allocator.allocate(count, dtype);
    if (this._data && newData) {
      const copyLen = Math.min(this._data.length, newData.length);
      for (let i = 0; i < copyLen; i++) newData[i] = this._data[i]!;
    }
    if (this._allocator && this._data) {
      this._allocator.free(this._data);
    }
    this._data = newData;
    this._nbytes = newNbytes;
  }

  clone(): StorageImpl {
    const allocator = getAllocator(this._device.type);
    let newData = null;
    if (this._data) {
      if (StorageImpl.#hostReadHook) StorageImpl.#hostReadHook(this._data);
      const Ctor = this._data.constructor as { new(length: number): NumericTypedArray };
      newData = new Ctor(this._data.length);
      (newData as NumericSettable).set(this._data as ArrayLike<number | bigint>);
    }
    return new StorageImpl(newData, this._nbytes, this._device, allocator);
  }
}
