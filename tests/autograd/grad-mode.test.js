import { describe, it, expect } from 'vitest';
import { GradMode, noGrad, enableGrad } from '../../src/autograd/grad_mode.js';

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
