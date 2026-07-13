import { DispatchKey } from './dispatch_key.js';
import type { DispatchKeyValue } from './dispatch_key.js';
import type { KernelFunction } from './boxing.js';

export class FallbackTable {
  private readonly _kernels: Array<KernelFunction | null>;

  constructor() {
    this._kernels = new Array(DispatchKey.NUM_KEYS).fill(null);
  }

  register(key: DispatchKeyValue, kernelFn: KernelFunction): void {
    this._kernels[key] = kernelFn;
  }

  remove(key: DispatchKeyValue): void {
    this._kernels[key] = null;
  }

  lookup(key: DispatchKeyValue): KernelFunction | null {
    return this._kernels[key];
  }

  has(key: DispatchKeyValue): boolean {
    return this._kernels[key] !== null;
  }

  registeredKeys(): DispatchKeyValue[] {
    const keys: DispatchKeyValue[] = [];
    for (let i = 0; i < this._kernels.length; i++) {
      if (this._kernels[i]) keys.push(i as DispatchKeyValue);
    }
    return keys;
  }
}
