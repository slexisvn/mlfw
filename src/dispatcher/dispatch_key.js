export const DispatchKey = Object.freeze({
  CPU: 0,
  GPU: 1,
  WASM: 2,
  META: 3,
  LAZY: 4,
  CUSTOM_0: 5,
  CUSTOM_1: 6,
  CUSTOM_2: 7,
  CUSTOM_3: 8,

  BATCHED: 20,
  VMAP: 24,
  FUNCTIONALIZE: 28,
  AUTOCAST: 32,

  AUTOGRAD: 40,
  AUTOGRAD_CPU: 41,
  AUTOGRAD_GPU: 42,
  AUTOGRAD_WASM: 43,

  TRACING: 48,

  NUM_KEYS: 49,
});

const _KEY_NAMES = new Array(DispatchKey.NUM_KEYS).fill(null);
for (const [name, val] of Object.entries(DispatchKey)) {
  if (name !== 'NUM_KEYS' && val < DispatchKey.NUM_KEYS) {
    _KEY_NAMES[val] = name;
  }
}

const _BACKEND_KEY_FOR_DEVICE = Object.freeze({
  cpu: DispatchKey.CPU,
  gpu: DispatchKey.GPU,
  wasm: DispatchKey.WASM,
  webgpu: DispatchKey.CUSTOM_0,
  meta: DispatchKey.META,
  lazy: DispatchKey.LAZY,
});

const _AUTOGRAD_KEY_FOR_BACKEND = Object.freeze({
  [DispatchKey.CPU]: DispatchKey.AUTOGRAD_CPU,
  [DispatchKey.GPU]: DispatchKey.AUTOGRAD_GPU,
  [DispatchKey.WASM]: DispatchKey.AUTOGRAD_WASM,
});

export function backendKeyForDevice(deviceType) {
  const k = _BACKEND_KEY_FOR_DEVICE[deviceType];
  if (k === undefined) throw new Error(`No backend key for device: ${deviceType}`);
  return k;
}

export function autogradKeyForBackend(backendKey) {
  return _AUTOGRAD_KEY_FOR_BACKEND[backendKey] ?? DispatchKey.AUTOGRAD;
}

export class DispatchKeySet {
  constructor(lo, hi) {
    this._lo = lo | 0;
    this._hi = hi | 0;
  }

  static fromKey(key) {
    if (key < 32) return new DispatchKeySet(1 << key, 0);
    return new DispatchKeySet(0, 1 << (key - 32));
  }

  static fromKeys(...keys) {
    let lo = 0, hi = 0;
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      if (k < 32) lo |= (1 << k);
      else hi |= (1 << (k - 32));
    }
    return new DispatchKeySet(lo, hi);
  }

  add(key) {
    if (key < 32) return new DispatchKeySet(this._lo | (1 << key), this._hi);
    return new DispatchKeySet(this._lo, this._hi | (1 << (key - 32)));
  }

  remove(key) {
    if (key < 32) return new DispatchKeySet(this._lo & ~(1 << key), this._hi);
    return new DispatchKeySet(this._lo, this._hi & ~(1 << (key - 32)));
  }

  has(key) {
    if (key < 32) return (this._lo & (1 << key)) !== 0;
    return (this._hi & (1 << (key - 32))) !== 0;
  }

  without(key) {
    return this.remove(key);
  }

  union(other) {
    return new DispatchKeySet(this._lo | other._lo, this._hi | other._hi);
  }

  intersect(other) {
    return new DispatchKeySet(this._lo & other._lo, this._hi & other._hi);
  }

  subtract(other) {
    return new DispatchKeySet(this._lo & ~other._lo, this._hi & ~other._hi);
  }

  isEmpty() {
    return this._lo === 0 && this._hi === 0;
  }

  equals(other) {
    return this._lo === other._lo && this._hi === other._hi;
  }

  highestPriority() {
    if (this._hi !== 0) {
      return (63 - Math.clz32(this._hi));
    }
    if (this._lo !== 0) {
      return (31 - Math.clz32(this._lo));
    }
    return -1;
  }

  lowestPriority() {
    if (this._lo !== 0) {
      return 31 - Math.clz32(this._lo & -this._lo);
    }
    if (this._hi !== 0) {
      return 32 + 31 - Math.clz32(this._hi & -this._hi);
    }
    return -1;
  }

  count() {
    return _popcount32(this._lo) + _popcount32(this._hi);
  }

  *[Symbol.iterator]() {
    let hi = this._hi;
    while (hi !== 0) {
      const bit = 31 - Math.clz32(hi);
      yield bit + 32;
      hi &= ~(1 << bit);
    }
    let lo = this._lo;
    while (lo !== 0) {
      const bit = 31 - Math.clz32(lo);
      yield bit;
      lo &= ~(1 << bit);
    }
  }

  toString() {
    const names = [];
    for (const k of this) {
      names.push(_KEY_NAMES[k] || String(k));
    }
    return `DispatchKeySet(${names.join(', ')})`;
  }
}

function _popcount32(n) {
  n = n - ((n >> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
  return (((n + (n >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

export const EMPTY_KEY_SET = new DispatchKeySet(0, 0);

export const BACKEND_KEY_SET = DispatchKeySet.fromKeys(
  DispatchKey.CPU, DispatchKey.GPU, DispatchKey.WASM,
  DispatchKey.META, DispatchKey.LAZY,
  DispatchKey.CUSTOM_0, DispatchKey.CUSTOM_1,
  DispatchKey.CUSTOM_2, DispatchKey.CUSTOM_3
);

export const AUTOGRAD_KEY_SET = DispatchKeySet.fromKeys(
  DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU,
  DispatchKey.AUTOGRAD_GPU, DispatchKey.AUTOGRAD_WASM
);

export const FUNCTIONALITY_KEY_SET = DispatchKeySet.fromKeys(
  DispatchKey.BATCHED, DispatchKey.VMAP,
  DispatchKey.FUNCTIONALIZE, DispatchKey.AUTOCAST,
  DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU,
  DispatchKey.AUTOGRAD_GPU, DispatchKey.AUTOGRAD_WASM,
  DispatchKey.TRACING
);
