import { EMPTY_KEY_SET } from './dispatch_key.js';
import type { DispatchKeySet } from './dispatch_key.js';

type GuardFrame = {
  exclude: DispatchKeySet;
  include: DispatchKeySet;
};

class GuardStack {
  private readonly _frames: GuardFrame[];

  constructor() {
    this._frames = [];
  }

  push(excludeKeys?: DispatchKeySet, includeKeys?: DispatchKeySet): void {
    this._frames.push({
      exclude: excludeKeys || EMPTY_KEY_SET,
      include: includeKeys || EMPTY_KEY_SET,
    });
  }

  pop(): GuardFrame | undefined {
    return this._frames.pop();
  }

  apply(keySet: DispatchKeySet): DispatchKeySet {
    let ks = keySet;
    for (let i = this._frames.length - 1; i >= 0; i--) {
      const frame = this._frames[i];
      ks = ks.subtract(frame.exclude);
      ks = ks.union(frame.include);
    }
    return ks;
  }

  get depth(): number {
    return this._frames.length;
  }

  clear(): void {
    this._frames.length = 0;
  }
}

export const guardStack = new GuardStack();

export function withExcludedKeys<T>(keys: DispatchKeySet, fn: () => T): T {
  guardStack.push(keys);
  try {
    return fn();
  } finally {
    guardStack.pop();
  }
}

type Thenable<T> = {
  then<TResult1 = T, TResult2 = never>(
    onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ): PromiseLike<TResult1 | TResult2>;
};

function isThenable<T>(value: T | Thenable<T>): value is Thenable<T> {
  return typeof value === 'object' && value !== null && 'then' in value && typeof value.then === 'function';
}

export function withIncludedKeys<T>(keys: DispatchKeySet, fn: () => T): T | PromiseLike<T> {
  guardStack.push(undefined, keys);
  let result: T;
  try {
    result = fn();
  } catch (error) {
    guardStack.pop();
    throw error;
  }
  if (isThenable(result)) {
    return result.then(
      value => { guardStack.pop(); return value; },
      error => { guardStack.pop(); throw error; },
    );
  }
  guardStack.pop();
  return result;
}

export function withGuard<T>(excludeKeys: DispatchKeySet, includeKeys: DispatchKeySet, fn: () => T): T {
  guardStack.push(excludeKeys, includeKeys);
  try {
    return fn();
  } finally {
    guardStack.pop();
  }
}
