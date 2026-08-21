import { StorageImpl } from './storage_impl.js';
import type { DType, NumericTypedArray } from '../types/dtype.js';
import type { Device } from '../types/device.js';

export class Storage {
  private readonly _impl: StorageImpl;

  constructor(impl: StorageImpl) {
    this._impl = impl;
  }

  static allocate(nbytes: number, dtype: DType, device: Device): Storage {
    return new Storage(StorageImpl.allocate(nbytes, dtype, device));
  }

  static fromData(data: NumericTypedArray | null, device: Device): Storage {
    return new Storage(StorageImpl.fromData(data, device));
  }

  get impl(): StorageImpl {
    return this._impl;
  }

  setPendingFill(fill: (() => void) | null) {
    this._impl.setPendingFill(fill);
  }

  get data(): NumericTypedArray | null {
    return this._impl.data;
  }

  get rawData(): NumericTypedArray | null {
    return this._impl.rawData;
  }

  get nbytes(): number {
    return this._impl.nbytes;
  }

  get device(): Device {
    return this._impl.device;
  }

  get isValid(): boolean {
    return this._impl.isValid;
  }

  get isMeta(): boolean {
    return this._impl.isMeta;
  }

  retain(): Storage {
    this._impl.retain();
    return this;
  }

  release() {
    this._impl.release();
  }

  clone(): Storage {
    return new Storage(this._impl.clone());
  }

  resize(nbytes: number, dtype: DType) {
    this._impl.resize(nbytes, dtype);
  }
}
