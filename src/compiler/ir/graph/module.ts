import { cloneGraphFunction } from './function.js';
import type { GraphFunction } from './function.js';

export class GraphModule {
  name: string;
  private _functions: Map<string, GraphFunction>;
  _version: number;

  constructor(name = 'module') {
    this.name = name;
    this._functions = new Map();
    this._version = 0;
  }

  get version(): number { return this._version; }

  addFunction(func: GraphFunction): GraphFunction {
    this._functions.set(func.name, func);
    func._module = this;
    this._version++;
    return func;
  }

  getFunction(name: string): GraphFunction | null {
    return this._functions.get(name) || null;
  }

  hasFunction(name: string): boolean {
    return this._functions.has(name);
  }

  removeFunction(name: string): boolean {
    const removed = this._functions.delete(name);
    if (removed) this._version++;
    return removed;
  }

  restoreFrom(snapshot: GraphModule): void {
    this._functions.clear();
    for (const func of snapshot) {
      this._functions.set(func.name, func);
      func._module = this;
    }
    this._version++;
  }

  get functionCount(): number { return this._functions.size; }

  functionNames(): string[] {
    return [...this._functions.keys()];
  }

  *functions(): Generator<GraphFunction, void, undefined> {
    yield* this._functions.values();
  }

  *[Symbol.iterator](): Generator<GraphFunction, void, undefined> {
    yield* this._functions.values();
  }

  verify(): string[] {
    const errors: string[] = [];
    if (this._functions.size === 0) {
      errors.push('Module has no functions');
    }
    for (const func of this._functions.values()) {
      const funcErrors = func.verify();
      for (let i = 0; i < funcErrors.length; i++) {
        errors.push(`${func.name}: ${funcErrors[i]}`);
      }
    }
    return errors;
  }
}

export function cloneGraphModule(module: GraphModule): GraphModule {
  const clone = new GraphModule(module.name);
  for (const func of module) {
    clone.addFunction(cloneGraphFunction(func));
  }
  clone._version = module._version;
  return clone;
}
