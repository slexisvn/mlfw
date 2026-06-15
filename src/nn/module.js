import { Parameter } from './parameter.js';


export class Module {
  constructor() {
    this._parameters = new Map();
    this._buffers = new Map();
    this._modules = new Map();
    this._training = true;
  }

  forward() {
    throw new Error(`${this.constructor.name}.forward() not implemented`);
  }

  call(...inputs) {
    return this.forward(...inputs);
  }

  registerParameter(name, param) {
    if (param !== null && !(param instanceof Parameter)) {
      throw new Error('Expected Parameter instance');
    }
    this._parameters.set(name, param);
  }

  registerBuffer(name, tensor) {
    this._buffers.set(name, tensor);
  }

  registerModule(name, module) {
    if (module !== null && !(module instanceof Module)) {
      throw new Error('Expected Module instance');
    }
    this._modules.set(name, module);
  }

  *parameters(recurse = true) {
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

  *namedParameters(prefix = '', recurse = true) {
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

  *buffers(recurse = true) {
    for (const [, b] of this._buffers) {
      if (b !== null) yield b;
    }
    if (recurse) {
      for (const [, m] of this._modules) {
        if (m !== null) yield* m.buffers(true);
      }
    }
  }

  *children() {
    this._autoDetect();
    for (const [, m] of this._modules) {
      if (m !== null) yield m;
    }
  }

  *namedChildren() {
    this._autoDetect();
    for (const [name, m] of this._modules) {
      if (m !== null) yield [name, m];
    }
  }

  *modules() {
    yield this;
    this._autoDetect();
    for (const [, m] of this._modules) {
      if (m !== null) yield* m.modules();
    }
  }

  *namedModules(prefix = '') {
    yield [prefix, this];
    this._autoDetect();
    const sep = prefix ? prefix + '.' : '';
    for (const [name, m] of this._modules) {
      if (m !== null) yield* m.namedModules(sep + name);
    }
  }

  stateDict(prefix = '') {
    this._autoDetect();
    const dict = new Map();
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

  loadStateDict(dict) {
    this._autoDetect();
    for (const [name, p] of this._parameters) {
      const key = name;
      if (dict.has(key) && p !== null) {
        const src = dict.get(key);
        const dstData = p._impl.storage.data;
        const srcData = src._impl ? src._impl.storage.data : src.data;
        for (let i = 0; i < dstData.length; i++) dstData[i] = srcData[i];
      }
    }
    for (const [name, m] of this._modules) {
      if (m !== null) {
        const subDict = new Map();
        const prefix = name + '.';
        for (const [k, v] of dict) {
          if (k.startsWith(prefix)) subDict.set(k.substring(prefix.length), v);
        }
        if (subDict.size > 0) m.loadStateDict(subDict);
      }
    }
  }

  train(mode = true) {
    this._training = mode;
    for (const [, m] of this._modules) {
      if (m !== null) m.train(mode);
    }
    return this;
  }

  eval() {
    return this.train(false);
  }

  get training() {
    return this._training;
  }

  to(device) {
    this._autoDetect();
    for (const [name, p] of this._parameters) {
      if (p !== null && typeof p.to === 'function') {
        this._parameters.set(name, new Parameter(p.to(device), p.requiresGrad));
        this[name] = this._parameters.get(name);
      }
    }
    for (const [name, b] of this._buffers) {
      if (b !== null && typeof b.to === 'function') {
        this._buffers.set(name, b.to(device));
        this[name] = this._buffers.get(name);
      }
    }
    for (const [, m] of this._modules) {
      if (m !== null) m.to(device);
    }
    return this;
  }

  apply(fn) {
    for (const [, m] of this._modules) {
      if (m !== null) m.apply(fn);
    }
    fn(this);
    return this;
  }

  zeroGrad() {
    for (const p of this.parameters()) {
      if (p.grad) {
        const data = p.grad._impl.storage.data;
        if (data) data.fill(0);
      }
    }
    return this;
  }

  _autoDetect() {
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

  toString() {
    return this._buildRepr('');
  }

  _buildRepr(indent) {
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
