import { EMPTY_KEY_SET } from './dispatch_key.js';
import type { DispatchKeySet, DispatchKeyValue } from './dispatch_key.js';
import { OperatorEntry } from './operator_entry.js';
import { OperatorHandle } from './operator_handle.js';
import { FallbackTable } from './fallback.js';
import { guardStack } from './guard.js';
import { _setDispatcher } from './library.js';
import { parseSchema } from './operator_schema.js';
import type { OperatorSchema } from './operator_schema.js';
import type { KernelFunction } from './boxing.js';

type DispatchArg = unknown;

type Dispatchable = {
  dispatchKeySet?: DispatchKeySet;
};

function hasDispatchKeySet(value: unknown): value is Dispatchable {
  return typeof value === 'object' && value !== null && 'dispatchKeySet' in value;
}

class Dispatcher {
  private readonly _entries: Map<string, OperatorEntry>;
  private readonly _handles: Map<string, OperatorHandle>;
  private readonly _fallbacks: FallbackTable;

  constructor() {
    this._entries = new Map();
    this._handles = new Map();
    this._fallbacks = new FallbackTable();
  }

  registerOp(schema: OperatorSchema): OperatorHandle {
    const key = schema.key();
    const existing = this._handles.get(key);
    if (existing) return existing;
    const entry = new OperatorEntry(schema);
    this._entries.set(key, entry);
    const handle = new OperatorHandle(entry, schema);
    this._handles.set(key, handle);
    return handle;
  }

  findOp(name: string): OperatorHandle | null {
    const fullName = name.includes('::') ? name : `mlc::${name}`;
    return this._handles.get(fullName) || null;
  }

  findOrRegisterOp(name: string): OperatorHandle {
    let handle = this.findOp(name);
    if (!handle) {
      const schema = parseSchema(`${name}() -> Tensor`, 'mlc');
      handle = this.registerOp(schema);
    }
    return handle;
  }

  registerKernel(name: string, key: DispatchKeyValue, kernelFn: KernelFunction): void {
    const fullName = name.includes('::') ? name : `mlc::${name}`;
    const entry = this._entries.get(fullName);
    if (!entry) throw new Error(`Op '${fullName}' not registered`);
    entry.registerKernel(key, kernelFn);
  }

  registerFallback(key: DispatchKeyValue, kernelFn: KernelFunction): void {
    this._fallbacks.register(key, kernelFn);
  }

  dispatch(handle: OperatorHandle, keySet: DispatchKeySet, ...args: DispatchArg[]): unknown {
    const ks = guardStack.apply(keySet);
    return this._dispatchInternal(handle, ks, args);
  }

  redispatch(handle: OperatorHandle, keySet: DispatchKeySet, ...args: DispatchArg[]): unknown {
    return this._dispatchInternal(handle, keySet, args);
  }

  _dispatchInternal(handle: OperatorHandle, keySet: DispatchKeySet, args: DispatchArg[]): unknown {
    const key = keySet.highestPriority();
    if (key < 0) {
      throw new Error(`No dispatch key found for op '${handle.name}'`);
    }

    let kernel = handle.lookupKernel(key as DispatchKeyValue);
    if (!kernel) {
      kernel = this._fallbacks.lookup(key as DispatchKeyValue);
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

    const remaining = keySet.without(key as DispatchKeyValue);
    return kernel.callUnboxed(remaining, ...args);
  }

  callOp(name: string, ...args: DispatchArg[]): unknown {
    const handle = this.findOp(name);
    if (!handle) throw new Error(`Op '${name}' not found`);
    const ks = computeKeySet(args, handle.schema);
    return this.dispatch(handle, ks, ...args);
  }

  listOps(): string[] {
    return [...this._handles.keys()];
  }

  hasOp(name: string): boolean {
    const fullName = name.includes('::') ? name : `mlc::${name}`;
    return this._handles.has(fullName);
  }

  get fallbacks(): FallbackTable {
    return this._fallbacks;
  }
}

function _unionArg(ks: DispatchKeySet, arg: unknown): DispatchKeySet {
  if (!arg) return ks;
  if (hasDispatchKeySet(arg) && arg.dispatchKeySet) return ks.union(arg.dispatchKeySet);
  if (Array.isArray(arg)) {
    for (let i = 0; i < arg.length; i++) ks = _unionArg(ks, arg[i]);
  }
  return ks;
}

export function computeKeySet(args: readonly unknown[], schema?: OperatorSchema | null): DispatchKeySet {
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
