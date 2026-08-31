import { describe, it, expect } from 'vitest';
import { CanonicalSum, symKey, proportionalTo } from '../../../src/compiler/analysis/canonical.js';
import { SymInt } from '../../../src/compiler/ir/sym_int.js';

const i = () => SymInt.var('i');
const j = () => SymInt.var('j');

function termsOf(sum) {
  const out = {};
  for (const term of sum.orderedTerms()) out[symKey(term.atom)] = term.coeff;
  return out;
}

describe('symKey', () => {
  it('gives equal keys to structurally equal expressions built separately', () => {
    expect(symKey(SymInt.add(SymInt.mul(i(), 4), j()))).toBe(symKey(SymInt.add(SymInt.mul(SymInt.var('i'), 4), SymInt.var('j'))));
  });

  it('ignores operand order for commutative ops only', () => {
    expect(symKey(SymInt.max(i(), j()))).toBe(symKey(SymInt.max(j(), i())));
    expect(symKey(SymInt.div(i(), 4))).not.toBe(symKey(SymInt.div(4, i())));
  });

  it('separates a variable from a constant of the same spelling', () => {
    expect(symKey(SymInt.var('4'))).not.toBe(symKey(4));
  });
});

describe('CanonicalSum.of', () => {
  it('collects repeated atoms into one coefficient', () => {
    const sum = CanonicalSum.of(SymInt.add(SymInt.add(i(), i()), SymInt.mul(i(), 3)));
    expect(termsOf(sum)).toEqual({ $i: 5 });
    expect(sum.offset).toBe(0);
  });

  it('cancels a term that appears on both sides of a subtraction', () => {
    const lhs = SymInt.add(SymInt.mul(i(), 4), j());
    const rhs = SymInt.add(SymInt.mul(SymInt.var('i'), 4), 4);
    const sum = CanonicalSum.of(SymInt.sub(lhs, rhs));
    expect(termsOf(sum)).toEqual({ $j: 1 });
    expect(sum.offset).toBe(-4);
  });

  it('folds a difference of equal expressions to zero', () => {
    const expr = SymInt.add(SymInt.mul(i(), 7), 3);
    const sum = CanonicalSum.of(SymInt.sub(expr, SymInt.add(SymInt.mul(SymInt.var('i'), 7), 3)));
    expect(sum.isZero).toBe(true);
  });

  it('distributes a constant factor over a sum', () => {
    const sum = CanonicalSum.of(SymInt.mul(SymInt.add(i(), 2), 3));
    expect(termsOf(sum)).toEqual({ $i: 3 });
    expect(sum.offset).toBe(6);
  });

  it('keeps a product of two unknowns as one atom', () => {
    const sum = CanonicalSum.of(SymInt.mul(i(), j()));
    expect(sum.terms.size).toBe(1);
    expect(sum.isConstant).toBe(false);
  });

  it('treats div, mod, max and min as opaque atoms', () => {
    for (const expr of [SymInt.div(i(), 4), SymInt.mod(i(), 4), SymInt.max(i(), j()), SymInt.min(i(), j())]) {
      const sum = CanonicalSum.of(expr);
      expect(sum.terms.size).toBe(1);
      expect(sum.offset).toBe(0);
    }
  });

  it('canonicalizes inside an opaque atom so equal subterms still cancel', () => {
    const inner = SymInt.sub(SymInt.add(i(), 5), 5);
    const sum = CanonicalSum.of(SymInt.sub(SymInt.mod(inner, 4), SymInt.mod(i(), 4)));
    expect(sum.isZero).toBe(true);
  });

  it('round-trips back to an expression with the same value', () => {
    const expr = SymInt.sub(SymInt.add(SymInt.mul(i(), 3), SymInt.mul(j(), 2)), 5);
    const rebuilt = CanonicalSum.of(expr).toExpr();
    for (const [a, b] of [[0, 0], [3, 7], [-2, 5]]) {
      const env = new Map([['i', a], ['j', b]]);
      expect(SymInt.evaluate(rebuilt, env)).toBe(SymInt.evaluate(expr, env));
    }
  });
});

describe('CanonicalSum.divideBy', () => {
  it('divides when every coefficient and the offset are multiples', () => {
    const sum = CanonicalSum.of(SymInt.add(SymInt.mul(i(), 8), 4));
    expect(termsOf(sum.divideBy(4))).toEqual({ $i: 2 });
    expect(sum.divideBy(4).offset).toBe(1);
  });

  it('refuses when one coefficient is not a multiple', () => {
    const sum = CanonicalSum.of(SymInt.add(SymInt.mul(i(), 8), SymInt.mul(j(), 3)));
    expect(sum.divideBy(4)).toBeNull();
  });

  it('refuses when the offset is not a multiple', () => {
    expect(CanonicalSum.of(SymInt.add(SymInt.mul(i(), 8), 3)).divideBy(4)).toBeNull();
  });
});

describe('proportionalTo', () => {
  it('finds the multiple and the leftover constant', () => {
    const query = CanonicalSum.of(SymInt.sub(i(), j()));
    const fact = CanonicalSum.of(SymInt.sub(SymInt.sub(i(), j()), 3));
    expect(proportionalTo(fact, query)).toEqual({ factor: 1, remainder: -3 });
  });

  it('handles a scaled fact', () => {
    const query = CanonicalSum.of(SymInt.sub(i(), j()));
    const fact = CanonicalSum.of(SymInt.add(SymInt.mul(SymInt.sub(i(), j()), 2), 6));
    expect(proportionalTo(fact, query)).toEqual({ factor: 2, remainder: 6 });
  });

  it('rejects a fact whose terms do not line up', () => {
    const query = CanonicalSum.of(SymInt.sub(i(), j()));
    expect(proportionalTo(CanonicalSum.of(SymInt.add(i(), j())), query)).toBeNull();
    expect(proportionalTo(CanonicalSum.of(i()), query)).toBeNull();
    expect(proportionalTo(CanonicalSum.of(5), query)).toBeNull();
  });

  it('rejects a constant query, which no fact can bound', () => {
    expect(proportionalTo(CanonicalSum.of(i()), CanonicalSum.of(3))).toBeNull();
  });
});
