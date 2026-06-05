import { parseSchema } from './operator_schema.js';
import { KernelFunction } from './boxing.js';

let _dispatcher = null;

export function _setDispatcher(d) {
  _dispatcher = d;
}

export class Library {
  constructor(namespace, kind) {
    this._namespace = namespace;
    this._kind = kind;
    this._registrations = [];
  }

  def(schemaStr) {
    const schema = parseSchema(schemaStr, this._namespace);
    if (_dispatcher) {
      _dispatcher.registerOp(schema);
    }
    this._registrations.push({ type: 'def', schema });
    return this;
  }

  impl(name, key, fn) {
    const kernelFn = fn instanceof KernelFunction ? fn : KernelFunction.fromUnboxed(fn);
    if (_dispatcher) {
      _dispatcher.registerKernel(
        `${this._namespace}::${name}`,
        key,
        kernelFn
      );
    }
    this._registrations.push({ type: 'impl', name, key, kernelFn });
    return this;
  }

  implBoxed(name, key, fn) {
    const kernelFn = KernelFunction.fromBoxed(fn);
    if (_dispatcher) {
      _dispatcher.registerKernel(
        `${this._namespace}::${name}`,
        key,
        kernelFn
      );
    }
    this._registrations.push({ type: 'impl', name, key, kernelFn });
    return this;
  }

  fallback(key, fn) {
    const kernelFn = fn instanceof KernelFunction ? fn : KernelFunction.fromBoxed(fn);
    if (_dispatcher) {
      _dispatcher.registerFallback(key, kernelFn);
    }
    this._registrations.push({ type: 'fallback', key, kernelFn });
    return this;
  }

  replay(dispatcher) {
    for (const reg of this._registrations) {
      if (reg.type === 'def') {
        dispatcher.registerOp(reg.schema);
      } else if (reg.type === 'impl') {
        dispatcher.registerKernel(
          `${this._namespace}::${reg.name}`,
          reg.key,
          reg.kernelFn
        );
      } else if (reg.type === 'fallback') {
        dispatcher.registerFallback(reg.key, reg.kernelFn);
      }
    }
  }
}
