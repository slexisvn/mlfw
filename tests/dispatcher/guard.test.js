import { describe, it, expect, afterEach } from 'vitest';
import { DispatchKey, DispatchKeySet } from '../../src/dispatcher/dispatch_key.js';
import { guardStack, withExcludedKeys, withIncludedKeys, withGuard } from '../../src/dispatcher/guard.js';

afterEach(() => { guardStack.clear(); });

describe('guardStack apply', () => {
  it('removes excluded keys from key set', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const exclude = DispatchKeySet.fromKey(DispatchKey.AUTOGRAD);

    guardStack.push(exclude);
    const result = guardStack.apply(ks);
    guardStack.pop();

    expect(result.has(DispatchKey.CPU)).toBe(true);
    expect(result.has(DispatchKey.AUTOGRAD)).toBe(false);
  });

  it('adds included keys to key set', () => {
    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    const include = DispatchKeySet.fromKey(DispatchKey.TRACING);

    guardStack.push(undefined, include);
    const result = guardStack.apply(ks);
    guardStack.pop();

    expect(result.has(DispatchKey.CPU)).toBe(true);
    expect(result.has(DispatchKey.TRACING)).toBe(true);
  });

  it('applies multiple frames in stack order', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    guardStack.push(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD));
    guardStack.push(undefined, DispatchKeySet.fromKey(DispatchKey.TRACING));

    const result = guardStack.apply(ks);

    guardStack.pop();
    guardStack.pop();

    expect(result.has(DispatchKey.CPU)).toBe(true);
    expect(result.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(result.has(DispatchKey.TRACING)).toBe(true);
  });
});

describe('withExcludedKeys', () => {
  it('excludes keys during callback and restores after', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    let inside;
    withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), () => {
      inside = guardStack.apply(ks);
    });
    expect(inside.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(guardStack.depth).toBe(0);
  });

  it('restores even when callback throws', () => {
    const before = guardStack.depth;
    try {
      withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.CPU), () => {
        throw new Error('boom');
      });
    } catch {}
    expect(guardStack.depth).toBe(before);
  });
});

describe('withIncludedKeys', () => {
  it('includes keys during callback and restores after', () => {
    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    let inside;
    withIncludedKeys(DispatchKeySet.fromKey(DispatchKey.TRACING), () => {
      inside = guardStack.apply(ks);
    });
    expect(inside.has(DispatchKey.TRACING)).toBe(true);
    expect(guardStack.depth).toBe(0);
  });
});

describe('withGuard', () => {
  it('excludes and includes simultaneously', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    let inside;
    withGuard(
      DispatchKeySet.fromKey(DispatchKey.AUTOGRAD),
      DispatchKeySet.fromKey(DispatchKey.TRACING),
      () => { inside = guardStack.apply(ks); }
    );
    expect(inside.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(inside.has(DispatchKey.TRACING)).toBe(true);
    expect(inside.has(DispatchKey.CPU)).toBe(true);
  });

  it('restores even when callback throws', () => {
    const before = guardStack.depth;
    try {
      withGuard(
        DispatchKeySet.fromKey(DispatchKey.CPU),
        DispatchKeySet.fromKey(DispatchKey.TRACING),
        () => { throw new Error('fail'); }
      );
    } catch {}
    expect(guardStack.depth).toBe(before);
  });
});

describe('nested withExcludedKeys', () => {
  it('inner exclude adds to outer exclude', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU, DispatchKey.AUTOGRAD);
    let innerResult;
    withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), () => {
      withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.GPU), () => {
        innerResult = guardStack.apply(ks);
      });
    });
    expect(innerResult.has(DispatchKey.CPU)).toBe(true);
    expect(innerResult.has(DispatchKey.GPU)).toBe(false);
    expect(innerResult.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(guardStack.depth).toBe(0);
  });
});

describe('outer exclude takes precedence over inner include', () => {
  it('outer exclude frame removes key even if inner include adds it', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    let result;
    withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), () => {
      withIncludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), () => {
        result = guardStack.apply(ks);
      });
    });
    expect(result.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(result.has(DispatchKey.CPU)).toBe(true);
  });
});

describe('inner exclude does not affect outer scope', () => {
  it('key is available outside inner withExcludedKeys', () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    let innerResult, outerResult;
    withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), () => {
      innerResult = guardStack.apply(ks);
    });
    outerResult = guardStack.apply(ks);
    expect(innerResult.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(outerResult.has(DispatchKey.AUTOGRAD)).toBe(true);
  });
});

describe('guard frames with an async body', () => {
  it('withExcludedKeys keeps the frame across an await', async () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const seen = [];
    await withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), async () => {
      seen.push(guardStack.apply(ks).has(DispatchKey.AUTOGRAD));
      await Promise.resolve();
      seen.push(guardStack.apply(ks).has(DispatchKey.AUTOGRAD));
      await new Promise(resolve => setTimeout(resolve, 0));
      seen.push(guardStack.apply(ks).has(DispatchKey.AUTOGRAD));
    });
    expect(seen).toEqual([false, false, false]);
    expect(guardStack.depth).toBe(0);
  });

  it('withIncludedKeys keeps the frame across an await', async () => {
    const ks = DispatchKeySet.fromKey(DispatchKey.CPU);
    const seen = [];
    await withIncludedKeys(DispatchKeySet.fromKey(DispatchKey.TRACING), async () => {
      seen.push(guardStack.apply(ks).has(DispatchKey.TRACING));
      await Promise.resolve();
      seen.push(guardStack.apply(ks).has(DispatchKey.TRACING));
    });
    expect(seen).toEqual([true, true]);
    expect(guardStack.depth).toBe(0);
  });

  it('withGuard keeps the frame across an await', async () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.AUTOGRAD);
    const seen = [];
    await withGuard(
      DispatchKeySet.fromKey(DispatchKey.AUTOGRAD),
      DispatchKeySet.fromKey(DispatchKey.TRACING),
      async () => {
        await Promise.resolve();
        const applied = guardStack.apply(ks);
        seen.push(applied.has(DispatchKey.AUTOGRAD), applied.has(DispatchKey.TRACING));
      }
    );
    expect(seen).toEqual([false, true]);
    expect(guardStack.depth).toBe(0);
  });

  it('pops the frame when the async body rejects', async () => {
    const before = guardStack.depth;
    await expect(withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.CPU), async () => {
      await Promise.resolve();
      throw new Error('boom');
    })).rejects.toThrow('boom');
    expect(guardStack.depth).toBe(before);
  });

  it('nests async frames without leaking the inner frame', async () => {
    const ks = DispatchKeySet.fromKeys(DispatchKey.CPU, DispatchKey.GPU, DispatchKey.AUTOGRAD);
    let inner;
    let outerAfterInner;
    await withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.AUTOGRAD), async () => {
      await withExcludedKeys(DispatchKeySet.fromKey(DispatchKey.GPU), async () => {
        await Promise.resolve();
        inner = guardStack.apply(ks);
      });
      outerAfterInner = guardStack.apply(ks);
    });
    expect(inner.has(DispatchKey.GPU)).toBe(false);
    expect(inner.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(outerAfterInner.has(DispatchKey.GPU)).toBe(true);
    expect(outerAfterInner.has(DispatchKey.AUTOGRAD)).toBe(false);
    expect(guardStack.depth).toBe(0);
  });

  it('does not convert a synchronous body into a promise', () => {
    const cpu = DispatchKeySet.fromKey(DispatchKey.CPU);
    const tracing = DispatchKeySet.fromKey(DispatchKey.TRACING);
    expect(withExcludedKeys(cpu, () => guardStack.depth)).toBe(1);
    expect(withIncludedKeys(tracing, () => guardStack.depth)).toBe(1);
    expect(withGuard(cpu, tracing, () => guardStack.depth)).toBe(1);
    expect(guardStack.depth).toBe(0);
  });
});
