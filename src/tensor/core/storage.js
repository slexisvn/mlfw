import { StorageImpl } from './storage_impl.js';

export class Storage {
  constructor(impl) {
    this._impl = impl;
  }

  static allocate(nbytes, dtype, device) {
    return new Storage(StorageImpl.allocate(nbytes, dtype, device));
  }

  static fromData(data, device) {
    return new Storage(StorageImpl.fromData(data, device));
  }

  get impl() {
    return this._impl;
  }

  get data() {
    return this._impl.data;
  }

  get nbytes() {
    return this._impl.nbytes;
  }

  get device() {
    return this._impl.device;
  }

  get isValid() {
    return this._impl.isValid;
  }

  get isMeta() {
    return this._impl.isMeta;
  }

  retain() {
    this._impl.retain();
    return this;
  }

  release() {
    this._impl.release();
  }

  clone() {
    return new Storage(this._impl.clone());
  }

  resize(nbytes, dtype) {
    this._impl.resize(nbytes, dtype);
  }
}
