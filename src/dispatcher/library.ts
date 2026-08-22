import { parseSchema } from './operator_schema.js';
import { KernelFunction } from './boxing.js';
import type { UnboxedFn } from './boxing.js';
import type { DispatchKeyValue } from './dispatch_key.js';

type OperatorSchemaLike = {
  key(): string;
};

type DispatcherLike = {
  registerOp(schema: OperatorSchemaLike): unknown;
  registerKernel(name: string, key: DispatchKeyValue, kernelFn: KernelFunction): void;
  registerFallback(key: DispatchKeyValue, kernelFn: KernelFunction): void;
};

type Registration =
  | { type: 'def'; schema: OperatorSchemaLike }
  | { type: 'impl'; name: string; key: DispatchKeyValue; kernelFn: KernelFunction }
  | { type: 'fallback'; key: DispatchKeyValue; kernelFn: KernelFunction };

type KernelInput = KernelFunction | ((...args: never[]) => unknown);
type BoxedKernelInput = (keySet: unknown, stack: unknown[]) => unknown;

let _dispatcher: DispatcherLike | null = null;

export function _setDispatcher(d: DispatcherLike): void {
  _dispatcher = d;
}

export class Library {
  private readonly _namespace: string;
  private readonly _registrations: Registration[];

  constructor(namespace: string, kind: string) {
    this._namespace = namespace;
    this._registrations = [];
  }

  def(schemaStr: string): this {
    const schema = parseSchema(schemaStr, this._namespace) as OperatorSchemaLike;
    if (_dispatcher) {
      _dispatcher.registerOp(schema);
    }
    this._registrations.push({ type: 'def', schema });
    return this;
  }

  impl(name: string, key: DispatchKeyValue, fn: KernelInput): this {
    const kernelFn = fn instanceof KernelFunction ? fn : KernelFunction.fromUnboxed(fn as UnboxedFn);
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

  implBoxed(name: string, key: DispatchKeyValue, fn: BoxedKernelInput): this {
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

  fallback(key: DispatchKeyValue, fn: KernelFunction | BoxedKernelInput): this {
    const kernelFn = fn instanceof KernelFunction ? fn : KernelFunction.fromBoxed(fn);
    if (_dispatcher) {
      _dispatcher.registerFallback(key, kernelFn);
    }
    this._registrations.push({ type: 'fallback', key, kernelFn });
    return this;
  }

  replay(dispatcher: DispatcherLike): void {
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
