import { EMPTY_KEY_SET } from './dispatch_key.js';
import { scoped } from '../util/scoped.js';
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

function withGuardFrame<T>(excludeKeys: DispatchKeySet | undefined, includeKeys: DispatchKeySet | undefined, fn: () => T): T {
  guardStack.push(excludeKeys, includeKeys);
  return scoped(fn, () => { guardStack.pop(); });
}

export function withExcludedKeys<T>(keys: DispatchKeySet, fn: () => T): T {
  return withGuardFrame(keys, undefined, fn);
}

export function withIncludedKeys<T>(keys: DispatchKeySet, fn: () => T): T {
  return withGuardFrame(undefined, keys, fn);
}

export function withGuard<T>(excludeKeys: DispatchKeySet, includeKeys: DispatchKeySet, fn: () => T): T {
  return withGuardFrame(excludeKeys, includeKeys, fn);
}
