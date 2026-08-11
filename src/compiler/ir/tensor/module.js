import { verifyIR, IRLevel } from '../verify.js';

export class TirModule {
  constructor(name = 'module') {
    this.name = name;
    this._functions = new Map();
    this._version = 0;
  }

  get version() { return this._version; }

  addFunction(primFunc) {
    this._functions.set(primFunc.name, primFunc);
    primFunc._module = this;
    this._version++;
    return primFunc;
  }

  getFunction(name) {
    return this._functions.get(name) || null;
  }

  hasFunction(name) {
    return this._functions.has(name);
  }

  replaceFunction(name, primFunc) {
    if (!this._functions.has(name)) {
      throw new Error(`TirModule '${this.name}' has no function '${name}' to replace`);
    }
    if (primFunc.name !== name) this._functions.delete(name);
    this._functions.set(primFunc.name, primFunc);
    primFunc._module = this;
    this._version++;
    return primFunc;
  }

  removeFunction(name) {
    const removed = this._functions.delete(name);
    if (removed) this._version++;
    return removed;
  }

  get functionCount() { return this._functions.size; }

  functionNames() {
    return [...this._functions.keys()];
  }

  *functions() {
    yield* this._functions.values();
  }

  *[Symbol.iterator]() {
    yield* this._functions.values();
  }

  verify() {
    const errors = [];
    if (this._functions.size === 0) errors.push('Module has no functions');
    for (const [name, func] of this._functions) {
      if (func.name !== name) errors.push(`Function registered as '${name}' but named '${func.name}'`);
      for (const message of verifyIR(IRLevel.TIR, func)) errors.push(`${func.name}: ${message}`);
    }
    return errors;
  }
}
