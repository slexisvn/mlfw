import { describe, it, expect } from 'vitest';
import { Analyzer } from '../../../src/compiler/analysis/analyzer.js';
import { SymInt } from '../../../src/compiler/analysis/sym_int.js';

describe('arith Analyzer: ConstIntBound', () => {
  it('bounds a constant exactly', () => {
    const a = new Analyzer();
    const b = a.constIntBound(7);
    expect(b.min).toBe(7);
    expect(b.max).toBe(7);
    expect(b.isConst()).toBe(true);
  });

  it('bounds a bound variable and arithmetic over it', () => {
    const a = new Analyzer().bind('i', 0, 7);
    const i = SymInt.var('i');
    expect(a.constIntBound(i)).toMatchObject({ min: 0, max: 7 });
    expect(a.constIntBound(SymInt.add(i, 3))).toMatchObject({ min: 3, max: 10 });
    expect(a.constIntBound(SymInt.mul(i, 2))).toMatchObject({ min: 0, max: 14 });
    expect(a.constIntBound(SymInt.sub(10, i))).toMatchObject({ min: 3, max: 10 });
  });

  it('mod by a positive constant is bounded to [0, b-1]', () => {
    const a = new Analyzer().bind('i', 0, 1000);
    expect(a.constIntBound(SymInt.mod(SymInt.var('i'), 4))).toMatchObject({ min: 0, max: 3 });
  });

  it('floordiv / ceildiv by a positive constant', () => {
    const a = new Analyzer().bind('i', 0, 7);
    expect(a.constIntBound(SymInt.div(SymInt.var('i'), 2))).toMatchObject({ min: 0, max: 3 });
    expect(a.constIntBound(SymInt.ceilDiv(SymInt.var('i'), 2))).toMatchObject({ min: 0, max: 4 });
  });

  it('unbound variable yields ±Infinity', () => {
    const a = new Analyzer();
    const b = a.constIntBound(SymInt.var('n'));
    expect(b.min).toBe(-Infinity);
    expect(b.max).toBe(Infinity);
  });
});

describe('arith Analyzer: canProve', () => {
  it('proves non-negativity and bounds it can derive', () => {
    const a = new Analyzer().bind('i', 0, 7);
    const i = SymInt.var('i');
    expect(a.canProveNonNegative(i)).toBe(true);
    expect(a.canProveLess(i, 8)).toBe(true);
    expect(a.canProveGreaterEqual(i, 0)).toBe(true);
  });

  it('is sound: returns false when it cannot prove', () => {
    const a = new Analyzer().bind('i', 0, 7);
    const i = SymInt.var('i');
    expect(a.canProveGreaterEqual(i, 5)).toBe(false);
    expect(a.canProveLess(i, 4)).toBe(false);
  });

  it('proves structural equality including commutativity', () => {
    const a = new Analyzer();
    expect(a.canProveEqual(SymInt.add(SymInt.var('x'), SymInt.var('y')), SymInt.add(SymInt.var('y'), SymInt.var('x')))).toBe(true);
  });

  it('bindShape derives [0, extent-1] per dim', () => {
    const a = new Analyzer().bindShape(new Map([['B', 16]]));
    expect(a.canProveLess(SymInt.var('B'), 16)).toBe(true);
    expect(a.canProveNonNegative(SymInt.var('B'))).toBe(true);
  });
});

describe('arith Analyzer: canonical differences', () => {
  it('cancels the shared term before bounding a difference', () => {
    const a = new Analyzer().bind('i', 0, 15).bind('j', 0, 3);
    const lhs = SymInt.add(SymInt.mul(SymInt.var('i'), 4), SymInt.var('j'));
    const rhs = SymInt.add(SymInt.mul(SymInt.var('i'), 4), 4);
    expect(a.boundOfDifference(lhs, rhs)).toMatchObject({ min: -4, max: -1 });
    expect(a.canProveLess(lhs, rhs)).toBe(true);
  });

  it('proves a comparison the interval alone cannot decide', () => {
    const a = new Analyzer().bind('i', 0, 15).bind('j', 0, 3);
    const lhs = SymInt.add(SymInt.mul(SymInt.var('i'), 4), SymInt.var('j'));
    const rhs = SymInt.add(SymInt.mul(SymInt.var('i'), 4), 4);
    expect(a.constIntBound(SymInt.sub(lhs, rhs)).max >= 0).toBe(true);
    expect(a.canProveLess(lhs, rhs)).toBe(true);
  });

  it('proves equality between expressions written differently', () => {
    const a = new Analyzer();
    const i = SymInt.var('i');
    const lhs = SymInt.add(SymInt.add(i, SymInt.var('j')), 3);
    const rhs = SymInt.add(SymInt.add(SymInt.var('j'), 3), SymInt.var('i'));
    expect(a.canProveEqual(lhs, rhs)).toBe(true);
    expect(a.canProveEqual(lhs, SymInt.add(rhs, 1))).toBe(false);
  });

  it('keeps the interval result when nothing cancels', () => {
    const a = new Analyzer().bind('i', 0, 7).bind('j', 0, 7);
    expect(a.boundOfDifference(SymInt.var('i'), SymInt.var('j'))).toMatchObject({ min: -7, max: 7 });
  });
});

describe('arith Analyzer: divisibility', () => {
  it('proves i*8 is divisible by 8 and by 4, but not by 16', () => {
    const a = new Analyzer();
    const scaled = SymInt.mul(SymInt.var('i'), 8);
    expect(a.canProveDivisible(scaled, 8)).toBe(true);
    expect(a.canProveDivisible(scaled, 4)).toBe(true);
    expect(a.canProveDivisible(scaled, 16)).toBe(false);
  });

  it('proves divisibility of a sum of multiples without knowing any bound', () => {
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 16), SymInt.mul(SymInt.var('j'), 4));
    expect(new Analyzer().canProveDivisible(expr, 4)).toBe(true);
  });

  it('refuses divisibility once a free term is added', () => {
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 8), SymInt.var('j'));
    expect(new Analyzer().canProveDivisible(expr, 8)).toBe(false);
  });

  it('uses a modular binding on a variable', () => {
    const a = new Analyzer().bindModular('j', 4, 0);
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 8), SymInt.var('j'));
    expect(a.canProveDivisible(expr, 4)).toBe(true);
  });

  it('returns the exact quotient when the division is provably exact', () => {
    const a = new Analyzer();
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 8), 4);
    const quotient = a.exactQuotient(expr, 4);
    for (const value of [-3, 0, 5]) {
      const env = new Map([['i', value]]);
      expect(SymInt.evaluate(quotient, env)).toBe(SymInt.evaluate(SymInt.div(expr, 4), env));
    }
  });

  it('returns no quotient when divisibility does not hold', () => {
    expect(new Analyzer().exactQuotient(SymInt.add(SymInt.mul(SymInt.var('i'), 8), 3), 4)).toBeNull();
  });
});

describe('arith Analyzer: assumptions', () => {
  it('turns an assumption into a bound on an otherwise unknown variable', () => {
    const a = new Analyzer();
    const n = SymInt.var('n');
    expect(a.constIntBound(n).min).toBe(-Infinity);
    const release = a.assumeNonNegative(SymInt.sub(n, 1));
    expect(a.constIntBound(n).min).toBe(1);
    release();
    expect(a.constIntBound(n).min).toBe(-Infinity);
  });

  it('proves a loop guard from the loop bounds it was given', () => {
    const a = new Analyzer();
    const i = SymInt.var('i');
    const n = SymInt.var('n');
    a.assumeNonNegative(i);
    a.assumeLess(i, n);
    expect(a.canProveLess(i, n)).toBe(true);
    expect(a.canProveNonNegative(i)).toBe(true);
    expect(a.canProveLess(SymInt.add(i, 1), n)).toBe(false);
  });

  it('chains two assumptions transitively', () => {
    const a = new Analyzer();
    const x = SymInt.var('x');
    const y = SymInt.var('y');
    const z = SymInt.var('z');
    a.assumeNonNegative(SymInt.sub(x, y));
    a.assumeNonNegative(SymInt.sub(y, z));
    expect(a.canProveGreaterEqual(x, z)).toBe(true);
    expect(a.canProveGreaterEqual(z, x)).toBe(false);
  });

  it('releases assumptions in any order without corrupting the fact set', () => {
    const a = new Analyzer();
    const n = SymInt.var('n');
    const releaseFirst = a.assumeNonNegative(SymInt.sub(n, 1));
    const releaseSecond = a.assumeNonNegative(SymInt.sub(n, 5));
    expect(a.constIntBound(n).min).toBe(5);
    releaseFirst();
    releaseSecond();
    expect(a.constIntBound(n).min).toBe(-Infinity);
    expect(a.canProveNonNegative(n)).toBe(false);
  });

  it('ignores a release called twice', () => {
    const a = new Analyzer();
    const n = SymInt.var('n');
    const outer = a.assumeNonNegative(SymInt.sub(n, 1));
    const inner = a.assumeNonNegative(SymInt.sub(SymInt.var('m'), 2));
    inner();
    inner();
    expect(a.constIntBound(n).min).toBe(1);
    outer();
    expect(a.constIntBound(n).min).toBe(-Infinity);
  });

  it('drops an assumption that says nothing', () => {
    const a = new Analyzer();
    a.assumeNonNegative(5);
    expect(a.constIntBound(SymInt.var('n')).min).toBe(-Infinity);
  });
});
