import { describe, it, expect } from 'vitest';
import {
  DispatchKey, DispatchKeySet, EMPTY_KEY_SET,
  backendKeyForDevice,
} from '../../src/dispatcher/dispatch_key.js';

describe('DispatchKeySet bitwise operations', () => {
  it('fromKey sets correct bit for low key (CPU=0)', () => {
    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    expect(ks.has(DispatchKey.CPU)).toBe(true);
    expect(ks.has(DispatchKey.GPU)).toBe(false);
    expect(ks.count()).toBe(1);
  });

  it('fromKey sets correct bit for high key (AUTOGRAD=40)', () => {
    const ks = DispatchKeySet.fromKey(DispatchKey.AUTOGRAD);
    expect(ks.has(DispatchKey.AUTOGRAD)).toBe(true);
    expect(ks.has(DispatchKey.CPU)).toBe(false);
    expect(ks.count()).toBe(1);
  });

  it('fromKeys constructs set spanning lo and hi words', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    expect(ks.has(DispatchKey.CPU)).toBe(true);
    expect(ks.has(DispatchKey.AUTOGRAD)).toBe(true);
    expect(ks.has(DispatchKey.GPU)).toBe(false);
    expect(ks.count()).toBe(2);
  });

  it('add inserts a key without removing existing ones', () => {
    const ks = DispatchKeySet.fromKey(DispatchKey.CPU).add(DispatchKey.AUTOGRAD_CPU);
    expect(ks.has(DispatchKey.CPU)).toBe(true);
    expect(ks.has(DispatchKey.AUTOGRAD_CPU)).toBe(true);
    expect(ks.count()).toBe(2);
  });

  it('remove clears only the specified key', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU).remove(DispatchKey.CPU);
    expect(ks.has(DispatchKey.CPU)).toBe(false);
    expect(ks.has(DispatchKey.GPU)).toBe(true);
  });

  it('union merges two sets', () => {
    const a = DispatchKeySet.fromKey(DispatchKey.CPU);
    const b = DispatchKeySet.fromKey(DispatchKey.AUTOGRAD);
    const u = a.union(b);
    expect(u.has(DispatchKey.CPU)).toBe(true);
    expect(u.has(DispatchKey.AUTOGRAD)).toBe(true);
    expect(u.count()).toBe(2);
  });

  it('intersect keeps only shared keys', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU);
    const b = DispatchKeySet.fromKeys(DispatchKey.GPU, DispatchKey.WASM);
    const i = a.intersect(b);
    expect(i.has(DispatchKey.GPU)).toBe(true);
    expect(i.has(DispatchKey.CPU)).toBe(false);
    expect(i.has(DispatchKey.WASM)).toBe(false);
    expect(i.count()).toBe(1);
  });

  it('subtract removes all keys present in other set', () => {
    const full = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU);
    const autogradKeys = DispatchKeySet.fromKeys(DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU);
    const result = full.subtract(autogradKeys);
    expect(result.has(DispatchKey.CPU)).toBe(true);
    expect(result.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(result.has(DispatchKey.AUTOGRAD_CPU)).toBe(false);
  });
});

describe('highestPriority', () => {
  it('returns highest numbered key', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    expect(ks.highestPriority()).toBe(DispatchKey.AUTOGRAD);
  });

  it('returns -1 for empty set', () => {
    expect(EMPTY_KEY_SET.highestPriority()).toBe(-1);
  });

  it('handles keys only in hi word', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.AUTOGRAD, DispatchKey.TRACING);
    expect(ks.highestPriority()).toBe(DispatchKey.TRACING);
  });

  it('handles keys only in lo word', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.WASM);
    expect(ks.highestPriority()).toBe(DispatchKey.WASM);
  });
});

describe('lowestPriority', () => {
  it('returns lowest numbered key', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.GPU, DispatchKey.AUTOGRAD);
    expect(ks.lowestPriority()).toBe(DispatchKey.GPU);
  });

  it('returns -1 for empty set', () => {
    expect(EMPTY_KEY_SET.lowestPriority()).toBe(-1);
  });
});

describe('iterator', () => {
  it('yields keys in descending priority order', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU, DispatchKey.AUTOGRAD);
    const keys = [...ks];
    expect(keys).toEqual([DispatchKey.AUTOGRAD, DispatchKey.GPU, DispatchKey.CPU]);
  });

  it('yields nothing for empty set', () => {
    expect([...EMPTY_KEY_SET]).toEqual([]);
  });
});

describe('equals and isEmpty', () => {
  it('two sets with same keys are equal', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const b = DispatchKeySet.fromKeys(DispatchKey.AUTOGRAD, DispatchKey.CPU);
    expect(a.equals(b)).toBe(true);
  });

  it('different sets are not equal', () => {
    const a = DispatchKeySet.fromKey(DispatchKey.CPU);
    const b = DispatchKeySet.fromKey(DispatchKey.GPU);
    expect(a.equals(b)).toBe(false);
  });

  it('empty set is empty', () => {
    expect(EMPTY_KEY_SET.isEmpty()).toBe(true);
    expect(DispatchKeySet.fromKey(DispatchKey.CPU).isEmpty()).toBe(false);
  });
});

describe('count (popcount)', () => {
  it('counts keys across both lo and hi words', () => {
    const ks = DispatchKeySet.fromKeys(
      DispatchKey.CPU, DispatchKey.GPU, DispatchKey.WASM,
      DispatchKey.AUTOGRAD, DispatchKey.AUTOGRAD_CPU, DispatchKey.TRACING
    );
    expect(ks.count()).toBe(6);
  });

  it('returns 0 for empty set', () => {
    expect(EMPTY_KEY_SET.count()).toBe(0);
  });
});

describe('set algebra identities', () => {
  it('A subtract A = empty', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    expect(a.subtract(a).isEmpty()).toBe(true);
  });

  it('A intersect empty = empty', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU);
    expect(a.intersect(EMPTY_KEY_SET).isEmpty()).toBe(true);
  });

  it('A union empty = A', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU);
    expect(a.union(EMPTY_KEY_SET).equals(a)).toBe(true);
  });

  it('union is commutative', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const b = DispatchKeySet.fromKeys(DispatchKey.GPU, DispatchKey.TRACING);
    expect(a.union(b).equals(b.union(a))).toBe(true);
  });

  it('intersect is commutative', () => {
    const a = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU, DispatchKey.AUTOGRAD);
    const b = DispatchKeySet.fromKeys(DispatchKey.GPU, DispatchKey.AUTOGRAD, DispatchKey.TRACING);
    expect(a.intersect(b).equals(b.intersect(a))).toBe(true);
  });
});

describe('iterator completeness', () => {
  it('yields exactly the keys in the set, count matches', () => {
    const keys = [DispatchKey.CPU, DispatchKey.WASM, DispatchKey.BATCHED, DispatchKey.AUTOGRAD, DispatchKey.TRACING];
    const ks = DispatchKeySet.fromKeys(...keys);
    const iterated = [...ks];
    expect(iterated.length).toBe(keys.length);
    for (const k of keys) {
      expect(iterated).toContain(k);
    }
  });

  it('iteration order is strictly descending', () => {
    const ks = DispatchKeySet.fromKeys(
      DispatchKey.CPU, DispatchKey.WASM, DispatchKey.AUTOCAST,
      DispatchKey.AUTOGRAD, DispatchKey.TRACING
    );
    const keys = [...ks];
    for (let i = 1; i < keys.length; i++) {
      expect(keys[i]).toBeLessThan(keys[i - 1]);
    }
  });
});

describe('without is alias for remove', () => {
  it('without produces same result as remove', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU);
    expect(ks.without(DispatchKey.CPU).equals(ks.remove(DispatchKey.CPU))).toBe(true);
  });
});

describe('add is idempotent', () => {
  it('adding same key twice does not change the set', () => {
    const a = DispatchKeySet.fromKey(DispatchKey.CPU);
    const b = a.add(DispatchKey.CPU);
    expect(a.equals(b)).toBe(true);
    expect(b.count()).toBe(1);
  });
});

describe('boundary key 31 (last in lo word)', () => {
  it('key 31 is stored in lo word and key 32 in hi word', () => {
    const lo = new DispatchKeySet(1 << 31, 0);
    const hi = new DispatchKeySet(0, 1);
    expect(lo.has(31)).toBe(true);
    expect(lo.has(32)).toBe(false);
    expect(hi.has(32)).toBe(true);
    expect(hi.has(31)).toBe(false);
  });
});

describe('backendKeyForDevice', () => {
  it('throws for unknown device', () => {
    expect(() => backendKeyForDevice('tpu')).toThrow();
  });
});
