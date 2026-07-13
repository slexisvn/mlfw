import { Module } from '../module.js';

export class Sequential extends Module {
  private _length: number;

  constructor(...modules: Module[]) {
    super();
    for (let i = 0; i < modules.length; i++) {
      this[String(i)] = modules[i];
      this.registerModule(String(i), modules[i]);
    }
    this._length = modules.length;
  }

  forward(input: unknown): unknown {
    let x = input;
    for (let i = 0; i < this._length; i++) {
      x = (this[String(i)] as Module).forward(x);
    }
    return x;
  }

  get length(): number {
    return this._length;
  }

  *[Symbol.iterator](): Generator<Module> {
    for (let i = 0; i < this._length; i++) {
      yield this[String(i)] as Module;
    }
  }

  push(module: Module): this {
    const idx = this._length;
    this[String(idx)] = module;
    this.registerModule(String(idx), module);
    this._length++;
    return this;
  }
}

export class ModuleList extends Module {
  private _list: Module[];

  constructor(modules?: Module[]) {
    super();
    this._list = [];
    if (modules) {
      for (let i = 0; i < modules.length; i++) {
        this._list.push(modules[i]);
        this.registerModule(String(i), modules[i]);
      }
    }
  }

  get length(): number {
    return this._list.length;
  }

  get(i: number): Module | undefined {
    return this._list[i];
  }

  push(module: Module): this {
    const idx = this._list.length;
    this._list.push(module);
    this.registerModule(String(idx), module);
    return this;
  }

  *[Symbol.iterator](): Generator<Module> {
    for (const m of this._list) yield m;
  }

  forward(): never {
    throw new Error('ModuleList does not implement forward()');
  }
}

export class ModuleDict extends Module {
  private _dict: Map<string, Module>;

  constructor(modules?: Record<string, Module>) {
    super();
    this._dict = new Map();
    if (modules) {
      for (const [key, mod] of Object.entries(modules)) {
        this._dict.set(key, mod);
        this.registerModule(key, mod);
      }
    }
  }

  get(key: string): Module | undefined {
    return this._dict.get(key);
  }

  set(key: string, module: Module): this {
    this._dict.set(key, module);
    this.registerModule(key, module);
    return this;
  }

  has(key: string): boolean {
    return this._dict.has(key);
  }

  get size(): number {
    return this._dict.size;
  }

  *keys(): Generator<string> {
    yield* this._dict.keys();
  }

  *values(): Generator<Module> {
    yield* this._dict.values();
  }

  *[Symbol.iterator](): Generator<[string, Module]> {
    yield* this._dict.entries();
  }

  forward(): never {
    throw new Error('ModuleDict does not implement forward()');
  }
}
