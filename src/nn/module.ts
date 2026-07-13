import { Parameter } from './parameter.js';
import type { Tensor } from '../tensor/core/tensor.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';
import type { Device } from '../tensor/types/device.js';

type StateDictValue = Tensor | { _impl?: { storage: { data: NumericTypedArray | null } }; data?: NumericTypedArray | null };
type MovableTensor = Tensor & { to?: (device: Device | string) => Tensor };
type NumericFillArray = Exclude<NumericTypedArray, BigInt64Array>;


export class Module {
  [key: string]: unknown;

  protected _parameters: Map<string, Parameter | null>;
  protected _buffers: Map<string, Tensor | null>;
  protected _modules: Map<string, Module | null>;
  protected _training: boolean;
  private _detected?: boolean;

  constructor() {
    this._parameters = new Map();
    this._buffers = new Map();
    this._modules = new Map();
    this._training = true;
  }

  forward(..._inputs: unknown[]): unknown {
    throw new Error(`${this.constructor.name}.forward() not implemented`);
  }

  call(...inputs: unknown[]): unknown {
    return this.forward(...inputs);
  }

  registerParameter(name: string, param: Parameter | null): void {
    if (param !== null && !(param instanceof Parameter)) {
      throw new Error('Expected Parameter instance');
    }
    this._parameters.set(name, param);
  }

  registerBuffer(name: string, tensor: Tensor | null): void {
    this._buffers.set(name, tensor);
  }

  registerModule(name: string, module: Module | null): void {
    if (module !== null && !(module instanceof Module)) {
      throw new Error('Expected Module instance');
    }
    this._modules.set(name, module);
  }

  *parameters(recurse = true): Generator<Parameter> {
    this._autoDetect();
    for (const [, p] of this._parameters) {
      if (p !== null) yield p;
    }
    if (recurse) {
      for (const [, m] of this._modules) {
        if (m !== null) yield* m.parameters(true);
      }
    }
  }

  *namedParameters(prefix = '', recurse = true): Generator<[string, Parameter]> {
    this._autoDetect();
    const sep = prefix ? prefix + '.' : '';
    for (const [name, p] of this._parameters) {
      if (p !== null) yield [sep + name, p];
    }
    if (recurse) {
      for (const [mName, m] of this._modules) {
        if (m !== null) yield* m.namedParameters(sep + mName, true);
      }
    }
  }

  *buffers(recurse = true): Generator<Tensor> {
    for (const [, b] of this._buffers) {
      if (b !== null) yield b;
    }
    if (recurse) {
      for (const [, m] of this._modules) {
        if (m !== null) yield* m.buffers(true);
      }
    }
  }

  *children(): Generator<Module> {
    this._autoDetect();
    for (const [, m] of this._modules) {
      if (m !== null) yield m;
    }
  }

  *namedChildren(): Generator<[string, Module]> {
    this._autoDetect();
    for (const [name, m] of this._modules) {
      if (m !== null) yield [name, m];
    }
  }

  *modules(): Generator<Module> {
    yield this;
    this._autoDetect();
    for (const [, m] of this._modules) {
      if (m !== null) yield* m.modules();
    }
  }

  *namedModules(prefix = ''): Generator<[string, Module]> {
    yield [prefix, this];
    this._autoDetect();
    const sep = prefix ? prefix + '.' : '';
    for (const [name, m] of this._modules) {
      if (m !== null) yield* m.namedModules(sep + name);
    }
  }

  stateDict(prefix = ''): Map<string, Tensor> {
    this._autoDetect();
    const dict = new Map<string, Tensor>();
    const sep = prefix ? prefix + '.' : '';
    for (const [name, p] of this._parameters) {
      if (p !== null) dict.set(sep + name, p);
    }
    for (const [name, b] of this._buffers) {
      if (b !== null) dict.set(sep + name, b);
    }
    for (const [name, m] of this._modules) {
      if (m !== null) {
        for (const [k, v] of m.stateDict(sep + name)) {
          dict.set(k, v);
        }
      }
    }
    return dict;
  }

  loadStateDict(dict: Map<string, StateDictValue>): void {
    this._autoDetect();
    for (const [name, p] of this._parameters) {
      const key = name;
      if (dict.has(key) && p !== null) {
        const src = dict.get(key)!;
        const dstData = p._impl.storage.data!;
        const srcData = src._impl ? src._impl.storage.data! : src.data!;
        for (let i = 0; i < dstData.length; i++) dstData[i] = srcData[i];
      }
    }
    for (const [name, m] of this._modules) {
      if (m !== null) {
        const subDict = new Map<string, StateDictValue>();
        const prefix = name + '.';
        for (const [k, v] of dict) {
          if (k.startsWith(prefix)) subDict.set(k.substring(prefix.length), v);
        }
        if (subDict.size > 0) m.loadStateDict(subDict);
      }
    }
  }

  train(mode = true): this {
    this._autoDetect();
    this._training = mode;
    for (const [, m] of this._modules) {
      if (m !== null) m.train(mode);
    }
    return this;
  }

  eval(): this {
    return this.train(false);
  }

  get training(): boolean {
    return this._training;
  }

  to(device: Device | string): this {
    this._autoDetect();
    for (const [name, p] of this._parameters) {
      const movable = p as MovableTensor | null;
      if (p !== null && typeof movable?.to === 'function') {
        this._parameters.set(name, new Parameter(movable.to(device), p.requiresGrad));
        this[name] = this._parameters.get(name);
      }
    }
    for (const [name, b] of this._buffers) {
      const movable = b as MovableTensor | null;
      if (b !== null && typeof movable?.to === 'function') {
        this._buffers.set(name, movable.to(device));
        this[name] = this._buffers.get(name);
      }
    }
    for (const [, m] of this._modules) {
      if (m !== null) m.to(device);
    }
    return this;
  }

  apply(fn: (module: Module) => void): this {
    this._autoDetect();
    for (const [, m] of this._modules) {
      if (m !== null) m.apply(fn);
    }
    fn(this);
    return this;
  }

  zeroGrad(): this {
    for (const p of this.parameters()) {
      if (p.grad) {
        const data = p.grad._impl.storage.data;
        if (data) (data as NumericFillArray).fill(0);
      }
    }
    return this;
  }

  _autoDetect(): void {
    if (this._detected) return;
    this._detected = true;
    const keys = Object.keys(this);
    for (const key of keys) {
      if (key.startsWith('_')) continue;
      const val = this[key];
      if (val instanceof Parameter && !this._parameters.has(key)) {
        this._parameters.set(key, val);
      } else if (val instanceof Module && !this._modules.has(key)) {
        this._modules.set(key, val);
      }
    }
  }

  toString(): string {
    return this._buildRepr('');
  }

  _buildRepr(indent: string): string {
    this._autoDetect();
    const name = this.constructor.name;
    if (this._modules.size === 0) return name + '()';
    const lines = [name + '('];
    for (const [mName, m] of this._modules) {
      if (m !== null) {
        lines.push(indent + '  (' + mName + '): ' + m._buildRepr(indent + '  '));
      }
    }
    lines.push(indent + ')');
    return lines.join('\n');
  }
}
