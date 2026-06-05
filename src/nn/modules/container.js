import { Module } from '../module.js';

export class Sequential extends Module {
  constructor(...modules) {
    super();
    for (let i = 0; i < modules.length; i++) {
      this[String(i)] = modules[i];
      this.registerModule(String(i), modules[i]);
    }
    this._length = modules.length;
  }

  forward(input) {
    let x = input;
    for (let i = 0; i < this._length; i++) {
      x = this[String(i)].forward(x);
    }
    return x;
  }

  get length() {
    return this._length;
  }

  *[Symbol.iterator]() {
    for (let i = 0; i < this._length; i++) {
      yield this[String(i)];
    }
  }

  push(module) {
    const idx = this._length;
    this[String(idx)] = module;
    this.registerModule(String(idx), module);
    this._length++;
    return this;
  }
}

export class ModuleList extends Module {
  constructor(modules) {
    super();
    this._list = [];
    if (modules) {
      for (let i = 0; i < modules.length; i++) {
        this._list.push(modules[i]);
        this.registerModule(String(i), modules[i]);
      }
    }
  }

  get length() {
    return this._list.length;
  }

  get(i) {
    return this._list[i];
  }

  push(module) {
    const idx = this._list.length;
    this._list.push(module);
    this.registerModule(String(idx), module);
    return this;
  }

  *[Symbol.iterator]() {
    for (const m of this._list) yield m;
  }

  forward() {
    throw new Error('ModuleList does not implement forward()');
  }
}

export class ModuleDict extends Module {
  constructor(modules) {
    super();
    this._dict = new Map();
    if (modules) {
      for (const [key, mod] of Object.entries(modules)) {
        this._dict.set(key, mod);
        this.registerModule(key, mod);
      }
    }
  }

  get(key) {
    return this._dict.get(key);
  }

  set(key, module) {
    this._dict.set(key, module);
    this.registerModule(key, module);
    return this;
  }

  has(key) {
    return this._dict.has(key);
  }

  get size() {
    return this._dict.size;
  }

  *keys() {
    yield* this._dict.keys();
  }

  *values() {
    yield* this._dict.values();
  }

  *[Symbol.iterator]() {
    yield* this._dict.entries();
  }

  forward() {
    throw new Error('ModuleDict does not implement forward()');
  }
}
