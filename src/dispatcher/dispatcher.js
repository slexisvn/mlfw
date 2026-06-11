import { DispatchKeySet, EMPTY_KEY_SET } from './dispatch_key.js';
import { OperatorEntry } from './operator_entry.js';
import { OperatorHandle } from './operator_handle.js';
import { FallbackTable } from './fallback.js';
import { guardStack } from './guard.js';
import { _setDispatcher } from './library.js';

class Dispatcher {
  constructor() {
    this._entries = new Map();
    this._handles = new Map();
    this._fallbacks = new FallbackTable();
  }

  registerOp(schema) {
    const key = schema.key();
    if (this._handles.has(key)) return this._handles.get(key);
    const entry = new OperatorEntry(schema);
    this._entries.set(key, entry);
    const handle = new OperatorHandle(entry, schema);
    this._handles.set(key, handle);
    return handle;
  }

  findOp(name) {
    const fullName = name.includes('::') ? name : `mlc::${name}`;
    return this._handles.get(fullName) || null;
  }

  findOrRegisterOp(name) {
    let handle = this.findOp(name);
    if (!handle) {
      const { parseSchema } = require('./operator_schema.js');
      const schema = parseSchema(`${name}() -> Tensor`, 'mlc');
      handle = this.registerOp(schema);
    }
    return handle;
  }

  registerKernel(name, key, kernelFn) {
    const fullName = name.includes('::') ? name : `mlc::${name}`;
    const entry = this._entries.get(fullName);
    if (!entry) throw new Error(`Op '${fullName}' not registered`);
    entry.registerKernel(key, kernelFn);
  }

  registerFallback(key, kernelFn) {
    this._fallbacks.register(key, kernelFn);
  }

  dispatch(handle, keySet, ...args) {
    const ks = guardStack.apply(keySet);
    return this._dispatchInternal(handle, ks, args);
  }

  redispatch(handle, keySet, ...args) {
    return this._dispatchInternal(handle, keySet, args);
  }

  _dispatchInternal(handle, keySet, args) {
    const key = keySet.highestPriority();
    if (key < 0) {
      throw new Error(`No dispatch key found for op '${handle.name}'`);
    }

    let kernel = handle.lookupKernel(key);
    if (!kernel) {
      kernel = this._fallbacks.lookup(key);
    }
    if (!kernel) {
      const catchAll = handle.entry.catchAll;
      if (catchAll) {
        kernel = catchAll;
      }
    }
    if (!kernel) {
      throw new Error(
        `No kernel registered for op '${handle.name}' with dispatch key ${key}`
      );
    }

    const remaining = keySet.without(key);
    return kernel.callUnboxed(remaining, ...args);
  }

  callOp(name, ...args) {
    const handle = this.findOp(name);
    if (!handle) throw new Error(`Op '${name}' not found`);
    const ks = computeKeySet(args, handle.schema);
    return this.dispatch(handle, ks, ...args);
  }

  listOps() {
    return [...this._handles.keys()];
  }

  hasOp(name) {
    const fullName = name.includes('::') ? name : `mlc::${name}`;
    return this._handles.has(fullName);
  }

  get fallbacks() {
    return this._fallbacks;
  }
}

function _unionArg(ks, arg) {
  if (!arg) return ks;
  if (arg.dispatchKeySet) return ks.union(arg.dispatchKeySet);
  if (Array.isArray(arg)) {
    for (let i = 0; i < arg.length; i++) ks = _unionArg(ks, arg[i]);
  }
  return ks;
}

export function computeKeySet(args, schema) {
  let ks = EMPTY_KEY_SET;
  if (schema) {
    const indices = schema.tensorArgIndices;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (idx < args.length) {
        ks = _unionArg(ks, args[idx]);
      }
    }
  } else {
    for (let i = 0; i < args.length; i++) {
      ks = _unionArg(ks, args[i]);
    }
  }
  return ks;
}

export const dispatcher = new Dispatcher();
_setDispatcher(dispatcher);
