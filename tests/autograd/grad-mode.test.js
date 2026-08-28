import { describe, it, expect, afterEach } from 'vitest';
import { GradMode, noGrad, enableGrad } from '../../src/autograd/grad_mode.js';
import { randn } from '../../src/index.js';

afterEach(() => { GradMode.setEnabled(true); });

describe('noGrad', () => {
  it('disables grad inside callback and restores after', () => {
    GradMode.setEnabled(true);
    let insideState;
    noGrad(() => { insideState = GradMode.isEnabled(); });
    expect(insideState).toBe(false);
    expect(GradMode.isEnabled()).toBe(true);
  });

  it('restores state even when callback throws', () => {
    GradMode.setEnabled(true);
    try { noGrad(() => { throw new Error('fail'); }); } catch {}
    expect(GradMode.isEnabled()).toBe(true);
  });

  it('returns callback result', () => {
    expect(noGrad(() => 42)).toBe(42);
  });

  it('nests correctly', () => {
    let inner;
    noGrad(() => {
      noGrad(() => { inner = GradMode.isEnabled(); });
    });
    expect(inner).toBe(false);
    expect(GradMode.isEnabled()).toBe(true);
  });
});

describe('enableGrad', () => {
  it('re-enables grad inside noGrad and restores after', () => {
    let insideEnable, afterEnable;
    noGrad(() => {
      enableGrad(() => { insideEnable = GradMode.isEnabled(); });
      afterEnable = GradMode.isEnabled();
    });
    expect(insideEnable).toBe(true);
    expect(afterEnable).toBe(false);
  });

  it('restores state even when callback throws', () => {
    GradMode.setEnabled(false);
    try { enableGrad(() => { throw new Error('fail'); }); } catch {}
    expect(GradMode.isEnabled()).toBe(false);
    GradMode.setEnabled(true);
  });
});

describe('noGrad with an async body', () => {
  it('keeps grad disabled across an await', async () => {
    GradMode.setEnabled(true);
    const seen = [];
    await noGrad(async () => {
      seen.push(GradMode.isEnabled());
      await Promise.resolve();
      seen.push(GradMode.isEnabled());
      await new Promise(resolve => setTimeout(resolve, 0));
      seen.push(GradMode.isEnabled());
    });
    expect(seen).toEqual([false, false, false]);
    expect(GradMode.isEnabled()).toBe(true);
  });

  it('does not build an autograd graph after an await', async () => {
    GradMode.setEnabled(true);
    const a = randn([2, 2]).requiresGrad_(true);
    const flags = [];
    await noGrad(async () => {
      flags.push(a.add(a).requiresGrad);
      await Promise.resolve();
      flags.push(a.add(a).requiresGrad);
    });
    expect(flags).toEqual([false, false]);
    expect(a.add(a).requiresGrad).toBe(true);
  });

  it('restores state when the async body rejects', async () => {
    GradMode.setEnabled(true);
    await expect(noGrad(async () => {
      await Promise.resolve();
      throw new Error('fail');
    })).rejects.toThrow('fail');
    expect(GradMode.isEnabled()).toBe(true);
  });

  it('resolves to the async body result', async () => {
    await expect(noGrad(async () => 42)).resolves.toBe(42);
  });

  it('nests async blocks without leaking the inner frame', async () => {
    GradMode.setEnabled(true);
    let innerAfterAwait;
    let outerAfterInner;
    await noGrad(async () => {
      await enableGrad(async () => {
        await Promise.resolve();
        innerAfterAwait = GradMode.isEnabled();
      });
      outerAfterInner = GradMode.isEnabled();
    });
    expect(innerAfterAwait).toBe(true);
    expect(outerAfterInner).toBe(false);
    expect(GradMode.isEnabled()).toBe(true);
  });

  it('does not convert a synchronous body into a promise', () => {
    GradMode.setEnabled(true);
    expect(noGrad(() => GradMode.isEnabled())).toBe(false);
    expect(enableGrad(() => 7)).toBe(7);
    expect(GradMode.isEnabled()).toBe(true);
  });
});

describe('enableGrad with an async body', () => {
  it('keeps grad enabled across an await inside noGrad', async () => {
    GradMode.setEnabled(true);
    const seen = [];
    await noGrad(async () => {
      await enableGrad(async () => {
        seen.push(GradMode.isEnabled());
        await Promise.resolve();
        seen.push(GradMode.isEnabled());
      });
      seen.push(GradMode.isEnabled());
    });
    expect(seen).toEqual([true, true, false]);
    expect(GradMode.isEnabled()).toBe(true);
  });

  it('restores state when the async body rejects', async () => {
    GradMode.setEnabled(false);
    await expect(enableGrad(async () => {
      await Promise.resolve();
      throw new Error('fail');
    })).rejects.toThrow('fail');
    expect(GradMode.isEnabled()).toBe(false);
  });
});
