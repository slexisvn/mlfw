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
