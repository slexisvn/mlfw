export class GraphModule {
  constructor(name = 'module') {
    this.name = name;
    this._functions = new Map();
    this._version = 0;
  }

  get version() { return this._version; }

  addFunction(func) {
    this._functions.set(func.name, func);
    this._version++;
    return func;
  }

  getFunction(name) {
    return this._functions.get(name) || null;
  }

  hasFunction(name) {
    return this._functions.has(name);
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
