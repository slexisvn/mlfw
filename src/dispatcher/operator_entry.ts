import { DispatchKey } from './dispatch_key.js';
import type { DispatchKeySet, DispatchKeyValue } from './dispatch_key.js';
import type { KernelFunction } from './boxing.js';
import type { OperatorSchema } from './operator_schema.js';

export type KernelLookup = Readonly<{
  key: DispatchKeyValue | -1;
  kernel: KernelFunction;
}>;

export class OperatorEntry {
  private readonly _schema: OperatorSchema;
  private readonly _kernels: Array<KernelFunction | null>;
  private _catchAll: KernelFunction | null;

  constructor(schema: OperatorSchema) {
    this._schema = schema;
    this._kernels = new Array(DispatchKey.NUM_KEYS).fill(null);
    this._catchAll = null;
  }

  get schema(): OperatorSchema {
    return this._schema;
  }

  registerKernel(key: DispatchKeyValue, kernelFn: KernelFunction): void {
    this._kernels[key] = kernelFn;
  }

  removeKernel(key: DispatchKeyValue): void {
    this._kernels[key] = null;
  }

  lookupKernel(key: DispatchKeyValue): KernelFunction | null {
    return this._kernels[key];
  }

  hasKernel(key: DispatchKeyValue): boolean {
    return this._kernels[key] !== null;
  }

  get catchAll(): KernelFunction | null {
    return this._catchAll;
  }

  setCatchAll(kernelFn: KernelFunction): void {
    this._catchAll = kernelFn;
  }

  bestKernel(keySet: DispatchKeySet): KernelLookup | null {
    for (const key of keySet) {
      const k = this._kernels[key];
      if (k) return { key, kernel: k };
    }
    if (this._catchAll) return { key: -1, kernel: this._catchAll };
    return null;
  }

  registeredKeys(): DispatchKeyValue[] {
    const keys: DispatchKeyValue[] = [];
    for (let i = 0; i < this._kernels.length; i++) {
      if (this._kernels[i]) keys.push(i as DispatchKeyValue);
    }
    return keys;
  }
}
