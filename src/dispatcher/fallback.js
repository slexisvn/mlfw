import { DispatchKey } from './dispatch_key.js';

export class FallbackTable {
  constructor() {
    this._kernels = new Array(DispatchKey.NUM_KEYS).fill(null);
  }

  register(key, kernelFn) {
    this._kernels[key] = kernelFn;
  }

  remove(key) {
    this._kernels[key] = null;
  }

  lookup(key) {
    return this._kernels[key];
  }

  has(key) {
    return this._kernels[key] !== null;
  }

  registeredKeys() {
    const keys = [];
    for (let i = 0; i < this._kernels.length; i++) {
      if (this._kernels[i]) keys.push(i);
    }
    return keys;
  }
}
