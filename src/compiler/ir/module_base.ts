export type ModuleMember = { name: string; _module?: unknown };

export abstract class IRModule<F extends ModuleMember> {
  name: string;
  protected readonly _functions: Map<string, F>;
  _version: number;

  constructor(name = 'module') {
    this.name = name;
    this._functions = new Map();
    this._version = 0;
  }

  get version(): number { return this._version; }

  addFunction(func: F): F {
    this._functions.set(func.name, func);
    func._module = this;
    this._version++;
    return func;
  }

  getFunction(name: string): F | null {
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

  get functionCount(): number { return this._functions.size; }

  functionNames(): string[] {
    return [...this._functions.keys()];
  }

  *functions(): Generator<F, void, undefined> {
    yield* this._functions.values();
  }

  *[Symbol.iterator](): Generator<F, void, undefined> {
    yield* this._functions.values();
  }

  abstract verify(): string[];
}
