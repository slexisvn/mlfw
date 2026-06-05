import { DispatchKey } from './dispatch_key.js';

export class OperatorEntry {
  constructor(schema) {
    this._schema = schema;
    this._kernels = new Array(DispatchKey.NUM_KEYS).fill(null);
    this._catchAll = null;
  }

  get schema() {
    return this._schema;
  }

  registerKernel(key, kernelFn) {
    this._kernels[key] = kernelFn;
  }

  removeKernel(key) {
    this._kernels[key] = null;
  }

  lookupKernel(key) {
    return this._kernels[key];
  }

  hasKernel(key) {
    return this._kernels[key] !== null;
  }

  get catchAll() {
    return this._catchAll;
  }

  setCatchAll(kernelFn) {
    this._catchAll = kernelFn;
  }

  bestKernel(keySet) {
    for (const key of keySet) {
      const k = this._kernels[key];
      if (k) return { key, kernel: k };
    }
    if (this._catchAll) return { key: -1, kernel: this._catchAll };
    return null;
  }

  registeredKeys() {
    const keys = [];
    for (let i = 0; i < this._kernels.length; i++) {
      if (this._kernels[i]) keys.push(i);
    }
    return keys;
  }
}
