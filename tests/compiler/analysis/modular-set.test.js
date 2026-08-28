import { describe, it, expect } from 'vitest';
import { ModularSet, modularSetOf } from '../../../src/compiler/analysis/modular_set.js';
import { SymInt } from '../../../src/compiler/analysis/sym_int.js';

const NOTHING_KNOWN = () => null;

function setOf(expr, known = {}) {
  return modularSetOf(expr, name => {
    const entry = known[name];
    return entry ? new ModularSet(entry[0], entry[1]) : null;
  });
}

describe('ModularSet', () => {
  it('normalizes the base into [0, coeff)', () => {
    const set = new ModularSet(4, -3);
    expect(set.coeff).toBe(4);
    expect(set.base).toBe(1);
    expect(set.contains(-3)).toBe(true);
    expect(set.contains(5)).toBe(true);
    expect(set.contains(6)).toBe(false);
  });

  it('treats coeff 0 as a single known value', () => {
    const set = ModularSet.exact(7);
    expect(set.isExact).toBe(true);
    expect(set.contains(7)).toBe(true);
    expect(set.contains(8)).toBe(false);
  });

  it('divisibleBy needs both the base and the stride to be multiples', () => {
    expect(new ModularSet(8, 0).divisibleBy(4)).toBe(true);
    expect(new ModularSet(8, 4).divisibleBy(4)).toBe(true);
    expect(new ModularSet(8, 2).divisibleBy(4)).toBe(false);
    expect(new ModularSet(6, 0).divisibleBy(4)).toBe(false);
    expect(ModularSet.exact(12).divisibleBy(4)).toBe(true);
    expect(ModularSet.exact(13).divisibleBy(4)).toBe(false);
  });

  it('addition keeps the gcd of the two strides', () => {
    const sum = new ModularSet(8, 3).add(new ModularSet(12, 5));
    expect(sum.coeff).toBe(4);
    expect(sum.base).toBe(0);
  });

  it('multiplication by an exact value scales the stride', () => {
    const product = ModularSet.exact(8).multiply(new ModularSet(1, 0));
    expect(product.coeff).toBe(8);
    expect(product.base).toBe(0);
    expect(product.divisibleBy(8)).toBe(true);
  });

  it('multiplying two strided sets keeps every cross term', () => {
    const product = new ModularSet(4, 2).multiply(new ModularSet(6, 3));
    for (let i = -3; i <= 3; i++) {
      for (let j = -3; j <= 3; j++) {
        expect(product.contains((2 + 4 * i) * (3 + 6 * j))).toBe(true);
      }
    }
  });

  it('floor division is only exact when the set is divisible', () => {
    expect(new ModularSet(8, 0).floorDivideBy(4)).toMatchObject({ coeff: 2, base: 0 });
    expect(new ModularSet(8, 2).floorDivideBy(4).coeff).toBe(1);
  });

  it('modulo collapses to a single value when the stride is a multiple of the divisor', () => {
    const set = new ModularSet(8, 5).moduloBy(4);
    expect(set.isExact).toBe(true);
    expect(set.base).toBe(1);
  });

  it('modulo keeps the gcd when the stride is not a multiple', () => {
    const set = new ModularSet(6, 5).moduloBy(4);
    for (let k = -4; k <= 4; k++) {
      const value = 5 + 6 * k;
      expect(set.contains(((value % 4) + 4) % 4)).toBe(true);
    }
  });

  it('intersectBound moves the endpoints onto the congruence class', () => {
    const aligned = new ModularSet(4, 1).intersectBound(0, 10);
    expect(aligned).toEqual({ min: 1, max: 9 });
  });
});

describe('modularSetOf', () => {
  it('reads an integer literal exactly and rejects a fraction', () => {
    expect(setOf(12).isExact).toBe(true);
    expect(setOf(12).base).toBe(12);
    expect(modularSetOf(1.5, NOTHING_KNOWN).coeff).toBe(1);
  });

  it('knows i*8 is a multiple of 8 for any i', () => {
    const set = setOf(SymInt.mul(SymInt.var('i'), 8));
    expect(set.divisibleBy(8)).toBe(true);
    expect(set.divisibleBy(4)).toBe(true);
    expect(set.divisibleBy(16)).toBe(false);
  });

  it('knows i*16 + j*4 is a multiple of 4 but not of 8', () => {
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 16), SymInt.mul(SymInt.var('j'), 4));
    expect(setOf(expr).divisibleBy(4)).toBe(true);
    expect(setOf(expr).divisibleBy(8)).toBe(false);
  });

  it('loses divisibility as soon as an unconstrained term is added', () => {
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 8), SymInt.var('j'));
    expect(setOf(expr).divisibleBy(8)).toBe(false);
  });

  it('uses what is known about a variable', () => {
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 8), SymInt.var('j'));
    expect(setOf(expr, { j: [4, 0] }).divisibleBy(4)).toBe(true);
  });

  it('folds (i*8) % 8 to zero and (i*8) // 8 back to a free stride', () => {
    const scaled = SymInt.mul(SymInt.var('i'), 8);
    expect(setOf(SymInt.mod(scaled, 8))).toMatchObject({ coeff: 0, base: 0 });
    expect(setOf(SymInt.div(scaled, 8))).toMatchObject({ coeff: 1, base: 0 });
  });

  it('gives up on a division it cannot prove exact', () => {
    const expr = SymInt.add(SymInt.mul(SymInt.var('i'), 8), 3);
    expect(setOf(SymInt.div(expr, 8)).coeff).toBe(1);
  });

  it('keeps the class through max and min when both sides agree', () => {
    const a = SymInt.mul(SymInt.var('i'), 4);
    const b = SymInt.mul(SymInt.var('j'), 4);
    expect(setOf(SymInt.max(a, b)).divisibleBy(4)).toBe(true);
    expect(setOf(SymInt.max(a, SymInt.var('k'))).divisibleBy(4)).toBe(false);
  });

  it('agrees with concrete evaluation on a random affine expression', () => {
    const expr = SymInt.add(SymInt.add(SymInt.mul(SymInt.var('i'), 12), SymInt.mul(SymInt.var('j'), 18)), 6);
    const set = setOf(expr);
    for (let i = -5; i <= 5; i++) {
      for (let j = -5; j <= 5; j++) {
        expect(set.contains(SymInt.evaluate(expr, new Map([['i', i], ['j', j]])))).toBe(true);
      }
    }
    expect(set.divisibleBy(6)).toBe(true);
  });
});
