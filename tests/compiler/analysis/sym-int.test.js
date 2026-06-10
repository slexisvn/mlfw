import { describe, it, expect } from 'vitest';
import { SymInt } from '../../../src/compiler/analysis/sym_int.js';

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
