import { teracCompile, teracInvoke, teracRelease } from '#io/terac';

import type { TeracHandle, TeracLocation } from '#io/terac';
import type { RuntimeModuleLike } from '../../compiler/support/config_types.js';

export type TeracEntry = { name: string; inputs: number; outputs: number };

export type TeracOptions = TeracLocation & {
  device?: string;
  optLevel?: number;
};

export const TERAC_DEFAULTS: Required<Pick<TeracOptions, 'device' | 'optLevel'>> = {
  device: 'cpu',
  optLevel: 3,
};

type TensorArg = { data: unknown; shape: readonly number[]; dtype: string };

const finalizer = new FinalizationRegistry<TeracHandle>((handle) => teracRelease(handle));

export class TeracRuntimeModule implements RuntimeModuleLike {
  mlir: string;
  device: string;
  entries: Map<string, TeracEntry>;
  handle: TeracHandle | null;
  executionPlan?: unknown;

  constructor(mlir: string, entries: readonly TeracEntry[], options: TeracOptions = {}) {
    const { device, optLevel } = { ...TERAC_DEFAULTS, ...options };
    this.mlir = mlir;
    this.device = device;
    this.entries = new Map(entries.map((entry) => [entry.name, entry]));
    this.handle = teracCompile(mlir, device, optLevel, options);
    finalizer.register(this, this.handle, this);
  }

  _entry(funcName: string): TeracEntry {
    const entry = this.entries.get(funcName);
    if (!entry) throw new Error(`terac: no function named ${funcName}`);
    return entry;
  }

  run(funcName: string, ...args: unknown[]): void {
    const entry = this._entry(funcName);
    if (!this.handle) throw new Error('terac: the module has been released');
    if (args.length !== entry.inputs + entry.outputs) {
      throw new Error(`terac: ${funcName} takes ${entry.inputs} tensors and returns ${entry.outputs}, but the call passes ${args.length} buffers`);
    }

    const buffers: ArrayBufferView[][] = [[], []];
    const shapes: number[][] = [[], []];
    for (let i = 0; i < args.length; i++) {
      const side = i < entry.inputs ? 0 : 1;
      const tensor = args[i] as TensorArg;
      if (!ArrayBuffer.isView(tensor.data)) {
        throw new Error(`terac: ${funcName} argument ${i} is not host memory`);
      }
      buffers[side].push(tensor.data as ArrayBufferView);
      for (const extent of tensor.shape) shapes[side].push(extent);
    }

    teracInvoke(this.handle, funcName, buffers[0], shapes[0], buffers[1], shapes[1]);
  }

  runAsync(funcName: string, ...args: unknown[]): Promise<void> {
    return Promise.resolve(this.run(funcName, ...args));
  }

  isAsync(): boolean {
    return false;
  }

  getKernelSource(funcName: string): string | null {
    return this.entries.has(funcName) ? this.mlir : null;
  }

  getKernelMetadata(funcName: string): Record<string, unknown> | null {
    const entry = this.entries.get(funcName);
    return entry ? { ...entry, device: this.device, compiler: 'terac' } : null;
  }

  listKernels(): string[] {
    return [...this.entries.keys()];
  }

  release(): void {
    if (!this.handle) return;
    finalizer.unregister(this);
    teracRelease(this.handle);
    this.handle = null;
  }
}
