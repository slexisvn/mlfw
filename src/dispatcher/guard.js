import { EMPTY_KEY_SET } from './dispatch_key.js';

class GuardStack {
  constructor() {
    this._frames = [];
  }

  push(excludeKeys, includeKeys) {
    this._frames.push({
      exclude: excludeKeys || EMPTY_KEY_SET,
      include: includeKeys || EMPTY_KEY_SET,
    });
  }

  pop() {
    return this._frames.pop();
  }

  apply(keySet) {
    let ks = keySet;
    for (let i = this._frames.length - 1; i >= 0; i--) {
      const frame = this._frames[i];
      ks = ks.subtract(frame.exclude);
      ks = ks.union(frame.include);
    }
    return ks;
  }

  get depth() {
    return this._frames.length;
  }

  clear() {
    this._frames.length = 0;
  }
}

export const guardStack = new GuardStack();

export function withExcludedKeys(keys, fn) {
  guardStack.push(keys, null);
  try {
    return fn();
  } finally {
    guardStack.pop();
  }
}

export function withIncludedKeys(keys, fn) {
  guardStack.push(null, keys);
  try {
    return fn();
  } finally {
    guardStack.pop();
  }
}

export function withGuard(excludeKeys, includeKeys, fn) {
  guardStack.push(excludeKeys, includeKeys);
  try {
    return fn();
  } finally {
    guardStack.pop();
  }
}
