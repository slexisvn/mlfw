import { KernelTable } from './kernel_table.js';
import type { DispatchKeySet, DispatchKeyValue } from './dispatch_key.js';
import type { KernelFunction } from './boxing.js';
import type { OperatorSchema } from './operator_schema.js';

export type KernelLookup = Readonly<{
  key: DispatchKeyValue | -1;
  kernel: KernelFunction;
}>;

export class OperatorEntry {
  private readonly _schema: OperatorSchema;
  private readonly _kernels: KernelTable;
  private _catchAll: KernelFunction | null;
  private _devices: ReadonlySet<string> | null;

  constructor(schema: OperatorSchema) {
    this._schema = schema;
    this._kernels = new KernelTable();
    this._catchAll = null;
    this._devices = null;
  }

  get schema(): OperatorSchema {
    return this._schema;
  }

  get devices(): ReadonlySet<string> | null {
    return this._devices;
  }

  declareDevices(devices: readonly string[]): void {
    this._devices = new Set(devices);
  }

  registerKernel(key: DispatchKeyValue, kernelFn: KernelFunction): void {
    this._kernels.register(key, kernelFn);
  }

  removeKernel(key: DispatchKeyValue): void {
    this._kernels.remove(key);
  }

  lookupKernel(key: DispatchKeyValue): KernelFunction | null {
    return this._kernels.lookup(key);
  }

  hasKernel(key: DispatchKeyValue): boolean {
    return this._kernels.has(key);
  }

  registeredKeys(): DispatchKeyValue[] {
    return this._kernels.registeredKeys();
  }

  get catchAll(): KernelFunction | null {
    return this._catchAll;
  }

  setCatchAll(kernelFn: KernelFunction): void {
    this._catchAll = kernelFn;
  }

  bestKernel(keySet: DispatchKeySet): KernelLookup | null {
    const found = this._kernels.firstOf(keySet);
    if (found) return found;
    if (this._catchAll) return { key: -1, kernel: this._catchAll };
    return null;
  }
}
