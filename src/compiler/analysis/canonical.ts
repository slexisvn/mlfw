import { SymInt, rebuildSym } from '../ir/sym_int.js';
import type { SymExpr } from '../ir/sym_int.js';

const COMMUTATIVE: ReadonlySet<string> = new Set(['add', 'mul', 'max', 'min']);

const _keyCache = new WeakMap<SymInt, string>();

export function symKey(expr: SymExpr): string {
  if (typeof expr === 'number') return String(expr);
  if (expr.type === 'var') return `$${expr.name as string}`;
  const cached = _keyCache.get(expr);
  if (cached !== undefined) return cached;
  const parts = expr.args.map(symKey);
  if (COMMUTATIVE.has(expr.type)) parts.sort();
  const key = `${expr.type}(${parts.join(',')})`;
  _keyCache.set(expr, key);
  return key;
}

export type CanonicalTerm = { coeff: number; atom: SymExpr };

export class CanonicalSum {
  offset: number;
  terms: Map<string, CanonicalTerm>;

  constructor(offset = 0, terms: Map<string, CanonicalTerm> = new Map()) {
    this.offset = offset;
    this.terms = terms;
  }

  static constant(value: number): CanonicalSum {
    return new CanonicalSum(value);
  }

  static atom(expr: SymExpr, coeff = 1): CanonicalSum {
    return new CanonicalSum().addTerm(coeff, expr);
  }

  static of(expr: SymExpr): CanonicalSum {
    if (typeof expr === 'number') return new CanonicalSum(expr);
    switch (expr.type) {
      case 'var':
        return CanonicalSum.atom(expr);
      case 'add':
        return new CanonicalSum().addSum(CanonicalSum.of(expr.args[0]), 1).addSum(CanonicalSum.of(expr.args[1]), 1);
      case 'sub':
        return new CanonicalSum().addSum(CanonicalSum.of(expr.args[0]), 1).addSum(CanonicalSum.of(expr.args[1]), -1);
      case 'neg':
        return CanonicalSum.of(expr.args[0]).scaled(-1);
      case 'mul': {
        const a = CanonicalSum.of(expr.args[0]);
        const b = CanonicalSum.of(expr.args[1]);
        if (a.isConstant) return b.scaled(a.offset);
        if (b.isConstant) return a.scaled(b.offset);
        return CanonicalSum.atom(SymInt.mul(a.toExpr(), b.toExpr()));
      }
      default: {
        const rebuilt = rebuildSym(expr.type, expr.args.map(arg => CanonicalSum.of(arg).toExpr()));
        return typeof rebuilt === 'number' ? new CanonicalSum(rebuilt) : CanonicalSum.atom(rebuilt);
      }
    }
  }

  get isConstant(): boolean {
    return this.terms.size === 0;
  }

  get isZero(): boolean {
    return this.terms.size === 0 && this.offset === 0;
  }

  addTerm(coeff: number, atom: SymExpr): this {
    if (coeff === 0) return this;
    const key = symKey(atom);
    const existing = this.terms.get(key);
    if (!existing) {
      this.terms.set(key, { coeff, atom });
      return this;
    }
    const next = existing.coeff + coeff;
    if (next === 0) this.terms.delete(key);
    else existing.coeff = next;
    return this;
  }

  addSum(other: CanonicalSum, scale: number): this {
    this.offset += other.offset * scale;
    for (const term of other.terms.values()) this.addTerm(term.coeff * scale, term.atom);
    return this;
  }

  scaled(factor: number): CanonicalSum {
    if (factor === 0) return new CanonicalSum(0);
    const out = new CanonicalSum(this.offset * factor);
    for (const [key, term] of this.terms) out.terms.set(key, { coeff: term.coeff * factor, atom: term.atom });
    return out;
  }

  plus(other: CanonicalSum): CanonicalSum {
    return new CanonicalSum().addSum(this, 1).addSum(other, 1);
  }

  minus(other: CanonicalSum): CanonicalSum {
    return new CanonicalSum().addSum(this, 1).addSum(other, -1);
  }

  orderedTerms(): CanonicalTerm[] {
    return [...this.terms.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).map(entry => entry[1]);
  }

  key(): string {
    const parts = this.orderedTerms().map(term => `${term.coeff}*${symKey(term.atom)}`);
    parts.push(String(this.offset));
    return parts.join('+');
  }

  toExpr(): SymExpr {
    let out: SymExpr = 0;
    for (const term of this.orderedTerms()) out = SymInt.add(out, SymInt.mul(term.coeff, term.atom));
    return SymInt.add(out, this.offset);
  }

  divideBy(divisor: number): CanonicalSum | null {
    if (!Number.isInteger(divisor) || divisor === 0) return null;
    if (this.offset % divisor !== 0) return null;
    const out = new CanonicalSum(this.offset / divisor);
    for (const [key, term] of this.terms) {
      if (term.coeff % divisor !== 0) return null;
      out.terms.set(key, { coeff: term.coeff / divisor, atom: term.atom });
    }
    return out;
  }
}

export function proportionalTo(fact: CanonicalSum, query: CanonicalSum): { factor: number; remainder: number } | null {
  if (query.isConstant || fact.terms.size !== query.terms.size) return null;
  let factor = 0;
  for (const [key, queryTerm] of query.terms) {
    const factTerm = fact.terms.get(key);
    if (!factTerm) return null;
    if (factTerm.coeff % queryTerm.coeff !== 0) return null;
    const ratio = factTerm.coeff / queryTerm.coeff;
    if (factor === 0) factor = ratio;
    else if (ratio !== factor) return null;
  }
  if (factor === 0) return null;
  return { factor, remainder: fact.offset - factor * query.offset };
}
