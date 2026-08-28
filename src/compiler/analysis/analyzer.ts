import { SymInt } from './sym_int.js';
import { CanonicalSum, proportionalTo, symKey } from './canonical.js';
import { ModularSet, modularSetOf } from './modular_set.js';
import type { CanonicalTerm } from './canonical.js';
import type { SymExpr } from './sym_int.js';

const NEG_INF = -Infinity;
const POS_INF = Infinity;

const MAX_DERIVED_FACTS = 24;

function isFinite2(x: number): boolean {
  return Number.isFinite(x);
}

function boundLo(x: number): number {
  return Number.isNaN(x) ? NEG_INF : x;
}

function boundHi(x: number): number {
  return Number.isNaN(x) ? POS_INF : x;
}

export class IntBound {
  min: number;
  max: number;

  constructor(min: number, max: number) {
    this.min = min;
    this.max = max;
  }
  isConst(): boolean {
    return this.min === this.max && isFinite2(this.min);
  }
}

const EVERYTHING = new IntBound(NEG_INF, POS_INF);

function decidesSign(bound: IntBound): boolean {
  return bound.min > 0 || bound.max < 0 || (bound.min === 0 && bound.max === 0);
}

export class Analyzer {
  private _varBounds: Map<string, IntBound>;
  private _varModular: Map<string, ModularSet>;
  private _stated: CanonicalSum[];
  private _derived: CanonicalSum[] | null;
  private _singleFacts: Map<string, CanonicalSum[]>;
  private _compoundSingleFacts: boolean;

  constructor() {
    this._varBounds = new Map();
    this._varModular = new Map();
    this._stated = [];
    this._derived = null;
    this._singleFacts = new Map();
    this._compoundSingleFacts = false;
  }

  bind(name: string, min: number, max: number): this {
    this._varBounds.set(name, new IntBound(min, max));
    return this;
  }

  bindModular(name: string, coeff: number, base: number): this {
    this._varModular.set(name, new ModularSet(coeff, base));
    return this;
  }

  getVarBound(name: string): IntBound | null {
    return this._varBounds.get(name) || null;
  }

  setVarBound(name: string, bound: IntBound | null): this {
    if (bound) this._varBounds.set(name, bound);
    else this._varBounds.delete(name);
    return this;
  }

  bindShape(shapeEnv: Iterable<[string, unknown]>): this {
    for (const [name, extent] of shapeEnv) {
      if (typeof extent === 'number' && extent > 0) this.bind(name, 0, extent - 1);
    }
    return this;
  }

  assumeNonNegative(expr: SymExpr): () => void {
    const sum = CanonicalSum.of(expr);
    if (sum.isConstant) return () => {};
    const depth = this._stated.length;
    this._stated.push(sum);
    this._derived = null;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this._stated.length = Math.min(this._stated.length, depth);
      this._derived = null;
    };
  }

  assumeLess(expr: SymExpr, limit: SymExpr): () => void {
    return this.assumeNonNegative(SymInt.sub(SymInt.sub(limit, expr), 1));
  }

  _facts(): readonly CanonicalSum[] {
    if (this._derived !== null) return this._derived;
    const facts = [...this._stated];
    for (let i = 0; i < this._stated.length && facts.length < MAX_DERIVED_FACTS; i++) {
      for (let j = i + 1; j < this._stated.length && facts.length < MAX_DERIVED_FACTS; j++) {
        const combined = this._stated[i].plus(this._stated[j]);
        if (!combined.isConstant) facts.push(combined);
      }
    }
    this._derived = facts;
    this._singleFacts = new Map();
    this._compoundSingleFacts = false;
    for (const fact of facts) {
      if (fact.terms.size !== 1) continue;
      const [key, term] = [...fact.terms.entries()][0];
      if (typeof term.atom !== 'number' && term.atom.type !== 'var') this._compoundSingleFacts = true;
      const bucket = this._singleFacts.get(key);
      if (bucket) bucket.push(fact);
      else this._singleFacts.set(key, [fact]);
    }
    return facts;
  }

  _singleFactBound(key: string): IntBound {
    const facts = this._singleFacts.get(key);
    if (!facts) return EVERYTHING;
    let min = NEG_INF;
    let max = POS_INF;
    for (const fact of facts) {
      const term = fact.terms.values().next().value as CanonicalTerm;
      if (term.coeff > 0) min = Math.max(min, Math.ceil(-fact.offset / term.coeff));
      else max = Math.min(max, Math.floor(-fact.offset / term.coeff));
    }
    return new IntBound(min, max);
  }

  modularSet(expr: SymExpr): ModularSet {
    return modularSetOf(expr, name => this._varModular.get(name) || null);
  }

  constIntBound(expr: SymExpr): IntBound {
    if (typeof expr === 'number') return new IntBound(expr, expr);
    if (!(expr instanceof SymInt)) return EVERYTHING;
    return this._atomBound(expr);
  }

  boundOfDifference(a: SymExpr, b: SymExpr): IntBound {
    const difference = SymInt.sub(a, b);
    const structural = this.constIntBound(difference);
    if (typeof difference === 'number' || decidesSign(structural)) return structural;
    const canonical = this._boundOfSum(CanonicalSum.of(difference));
    return new IntBound(Math.max(structural.min, canonical.min), Math.min(structural.max, canonical.max));
  }

  _boundOfSum(sum: CanonicalSum): IntBound {
    let min = sum.offset;
    let max = sum.offset;
    for (const term of sum.terms.values()) {
      const atom = this._atomBound(term.atom);
      const lo = term.coeff > 0 ? term.coeff * atom.min : term.coeff * atom.max;
      const hi = term.coeff > 0 ? term.coeff * atom.max : term.coeff * atom.min;
      min = boundLo(min + lo);
      max = boundHi(max + hi);
    }
    return this._tighten(sum, new IntBound(min, max));
  }

  _tighten(sum: CanonicalSum, bound: IntBound): IntBound {
    if (this._stated.length === 0) return bound;
    const facts = this._factBound(sum);
    const min = Math.max(bound.min, facts.min);
    const max = Math.min(bound.max, facts.max);
    return min > max ? bound : new IntBound(min, max);
  }

  _factBound(query: CanonicalSum): IntBound {
    if (query.isConstant) return EVERYTHING;
    let min = NEG_INF;
    let max = POS_INF;
    for (const fact of this._facts()) {
      const match = proportionalTo(fact, query);
      if (!match) continue;
      if (match.factor > 0) min = Math.max(min, Math.ceil(-match.remainder / match.factor));
      else max = Math.min(max, Math.floor(-match.remainder / match.factor));
    }
    return new IntBound(min, max);
  }

  _atomBound(atom: SymExpr): IntBound {
    if (typeof atom === 'number') return new IntBound(atom, atom);
    if (!(atom instanceof SymInt)) return EVERYTHING;

    const structural = atom.type === 'var'
      ? this._varBounds.get(atom.name as string) || EVERYTHING
      : this._structuralBound(atom);

    if (this._stated.length === 0) return structural;
    if (this._derived === null) this._facts();
    if (this._singleFacts.size === 0) return structural;
    const isVar = atom.type === 'var';
    if (!isVar && !this._compoundSingleFacts) return structural;
    const facts = this._singleFactBound(isVar ? `$${atom.name as string}` : symKey(atom));
    const min = Math.max(structural.min, facts.min);
    const max = Math.min(structural.max, facts.max);
    return min > max ? structural : new IntBound(min, max);
  }

  _structuralBound(expr: SymInt): IntBound {
    const a = this.constIntBound(expr.args[0]);
    if (expr.type === 'neg') return new IntBound(-a.max, -a.min);

    const b = this.constIntBound(expr.args[1]);
    switch (expr.type) {
      case 'add':
        return new IntBound(boundLo(a.min + b.min), boundHi(a.max + b.max));
      case 'sub':
        return new IntBound(boundLo(a.min - b.max), boundHi(a.max - b.min));
      case 'mul':
        return this._mulBound(a, b);
      case 'max':
        return new IntBound(Math.max(a.min, b.min), Math.max(a.max, b.max));
      case 'min':
        return new IntBound(Math.min(a.min, b.min), Math.min(a.max, b.max));
      case 'div':
        return this._divBound(a, b, Math.floor);
      case 'ceildiv':
        return this._divBound(a, b, Math.ceil);
      case 'mod':
        return this._modBound(a, b);
      default:
        return EVERYTHING;
    }
  }

  _mulBound(a: IntBound, b: IntBound): IntBound {
    if (!isFinite2(a.min) || !isFinite2(a.max) || !isFinite2(b.min) || !isFinite2(b.max)) {
      return EVERYTHING;
    }
    const products = [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max];
    return new IntBound(Math.min(...products), Math.max(...products));
  }

  _divBound(a: IntBound, b: IntBound, rounder: (x: number) => number): IntBound {
    if (b.isConst() && b.min > 0) {
      const lo = isFinite2(a.min) ? rounder(a.min / b.min) : a.min;
      const hi = isFinite2(a.max) ? rounder(a.max / b.min) : a.max;
      return new IntBound(lo, hi);
    }
    return EVERYTHING;
  }

  _modBound(a: IntBound, b: IntBound): IntBound {
    if (b.isConst() && b.min > 0) {
      return new IntBound(0, b.min - 1);
    }
    return EVERYTHING;
  }

  canProveGreaterEqual(expr: SymExpr, value: SymExpr): boolean {
    return this.boundOfDifference(expr, value).min >= 0;
  }

  canProveLess(expr: SymExpr, value: SymExpr): boolean {
    return this.boundOfDifference(expr, value).max < 0;
  }

  canProveNonNegative(expr: SymExpr): boolean {
    return this.canProveGreaterEqual(expr, 0);
  }

  canProveEqual(a: SymExpr, b: SymExpr): boolean {
    if (SymInt.equals(a, b)) return true;
    const bound = this.boundOfDifference(a, b);
    return bound.min === 0 && bound.max === 0;
  }

  canProveDivisible(expr: SymExpr, divisor: number): boolean {
    if (!Number.isInteger(divisor) || divisor <= 0) return false;
    if (divisor === 1) return true;
    if (this.modularSet(expr).divisibleBy(divisor)) return true;
    const bound = this.constIntBound(SymInt.mod(expr, divisor));
    return bound.min === 0 && bound.max === 0;
  }

  exactQuotient(expr: SymExpr, divisor: number): SymExpr | null {
    if (!Number.isInteger(divisor) || divisor <= 0) return null;
    if (!this.modularSet(expr).divisibleBy(divisor)) return null;
    const quotient = CanonicalSum.of(expr).divideBy(divisor);
    return quotient === null ? null : quotient.toExpr();
  }
}
