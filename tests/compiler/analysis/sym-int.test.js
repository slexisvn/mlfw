import { describe, it, expect } from 'vitest';
import { SymInt } from '../../../src/compiler/ir/sym_int.js';

describe('SymInt — division/modulo zero guards', () => {
  it('div(numeric, 0) throws', () => {
    expect(() => SymInt.div(10, 0)).toThrow();
  });

  it('mod(numeric, 0) throws', () => {
    expect(() => SymInt.mod(10, 0)).toThrow();
  });

  it('div(symbolic, 0) throws before constructing the node', () => {
    const x = SymInt.var('x');
    expect(() => SymInt.div(x, 0)).toThrow();
  });

  it('mod(symbolic, 0) throws before constructing the node', () => {
    const x = SymInt.var('x');
    expect(() => SymInt.mod(x, 0)).toThrow();
  });
});

describe('SymInt — substitute folds ceildiv when args become numeric', () => {
  it('substitute into ceildiv with all-numeric args yields a constant', () => {
    const n = SymInt.var('n');
    const expr = SymInt.ceilDiv(n, 4);
    const out = SymInt.substitute(expr, 'n', 10);
    expect(out).toBe(3);
  });

  it('substitute into ceildiv with one symbolic arg keeps the expression', () => {
    const n = SymInt.var('n');
    const m = SymInt.var('m');
    const expr = SymInt.ceilDiv(n, m);
    const out = SymInt.substitute(expr, 'n', 10);
    expect(out).toBeInstanceOf(SymInt);
    expect(out.type).toBe('ceildiv');
  });

  it('evaluate ceildiv produces ceiling semantics', () => {
    const n = SymInt.var('n');
    const expr = SymInt.ceilDiv(n, 3);
    expect(SymInt.evaluate(expr, new Map([['n', 10]]))).toBe(4);
    expect(SymInt.evaluate(expr, new Map([['n', 9]]))).toBe(3);
  });
});

describe('SymInt — division cancels the factors it can prove away', () => {
  const n = SymInt.var('n');
  const m = SymInt.var('m');

  it('cancels a symbol that appears in both products', () => {
    expect(SymInt.equals(SymInt.div(SymInt.mul(n, 3), n), 3)).toBe(true);
    expect(SymInt.equals(SymInt.div(SymInt.mul(SymInt.mul(n, m), 4), SymInt.mul(m, 4)), n)).toBe(true);
  });

  it('cancels a constant factor only when it divides evenly', () => {
    expect(SymInt.equals(SymInt.div(SymInt.mul(SymInt.mul(n, 2), 10), 20), n)).toBe(true);
    const partial = SymInt.div(SymInt.mul(n, 2), 4);
    expect(partial).toBeInstanceOf(SymInt);
    expect(partial.type).toBe('div');
  });

  it('leaves an expression it cannot cancel alone', () => {
    const kept = SymInt.div(SymInt.mul(n, 3), m);
    expect(kept.type).toBe('div');
    expect(SymInt.equals(kept.args[0], SymInt.mul(n, 3))).toBe(true);
    expect(SymInt.equals(kept.args[1], m)).toBe(true);
  });

  it('agrees with evaluation on every simplification it makes', () => {
    const env = new Map([['n', 7], ['m', 5]]);
    for (const [num, den] of [
      [SymInt.mul(n, 3), n],
      [SymInt.mul(SymInt.mul(n, m), 4), SymInt.mul(m, 4)],
      [SymInt.mul(SymInt.mul(n, 2), 10), 20],
      [SymInt.mul(n, 2), 4],
      [SymInt.mul(n, 3), m],
    ]) {
      const simplified = SymInt.div(num, den);
      const literal = new SymInt('div', null, [num, den]);
      expect(SymInt.evaluate(simplified, env)).toBe(SymInt.evaluate(literal, env));
    }
  });
});
