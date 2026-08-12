import { verifyIR, IRLevel } from '../verify.js';
import type { PrimFunc } from './nodes.js';

export class TirModule {
  name: string;
  private _functions: Map<string, PrimFunc>;
  _version: number;

  constructor(name = 'module') {
    this.name = name;
    this._functions = new Map();
    this._version = 0;
  }

  get version(): number { return this._version; }

  addFunction(primFunc: PrimFunc): PrimFunc {
    this._functions.set(primFunc.name, primFunc);
    primFunc._module = this;
    this._version++;
    return primFunc;
  }

  getFunction(name: string): PrimFunc | null {
    return this._functions.get(name) || null;
  }

  hasFunction(name: string): boolean {
    return this._functions.has(name);
  }

  replaceFunction(name: string, primFunc: PrimFunc): PrimFunc {
    if (!this._functions.has(name)) {
      throw new Error(`TirModule '${this.name}' has no function '${name}' to replace`);
    }
    if (primFunc.name !== name) this._functions.delete(name);
    this._functions.set(primFunc.name, primFunc);
    primFunc._module = this;
    this._version++;
    return primFunc;
  }

  removeFunction(name: string): boolean {
    const removed = this._functions.delete(name);
    if (removed) this._version++;
    return removed;
  }

  get functionCount(): number { return this._functions.size; }

  functionNames(): string[] {
    return [...this._functions.keys()];
  }

  *functions(): Generator<PrimFunc, void, undefined> {
    yield* this._functions.values();
  }

  *[Symbol.iterator](): Generator<PrimFunc, void, undefined> {
    yield* this._functions.values();
  }

  verify(): string[] {
    const errors: string[] = [];
    if (this._functions.size === 0) errors.push('Module has no functions');
    for (const [name, func] of this._functions) {
      if (func.name !== name) errors.push(`Function registered as '${name}' but named '${func.name}'`);
      for (const message of verifyIR(IRLevel.TIR, func)) errors.push(`${func.name}: ${message}`);
    }
    return errors;
  }
}
