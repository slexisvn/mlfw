import { gcd } from '../../util/integer.js';
import { floorMod } from '../../util/divmod.js';
import { SymInt } from './sym_int.js';
import type { SymExpr } from './sym_int.js';

export type ModularLookup = (name: string) => ModularSet | null;

export class ModularSet {
  readonly coeff: number;
  readonly base: number;

  constructor(coeff: number, base: number) {
    const c = Math.abs(coeff);
    this.coeff = c;
    this.base = c === 0 ? base : floorMod(base, c);
  }

  static exact(value: number): ModularSet {
    return new ModularSet(0, value);
  }

  static everything(): ModularSet {
    return EVERYTHING;
  }

  get isExact(): boolean {
    return this.coeff === 0;
  }

  divisibleBy(divisor: number): boolean {
    if (!Number.isInteger(divisor) || divisor <= 0) return false;
    if (floorMod(this.base, divisor) !== 0) return false;
    return this.isExact || this.coeff % divisor === 0;
  }

  contains(value: number): boolean {
    return this.isExact ? this.base === value : floorMod(value, this.coeff) === this.base;
  }

  add(other: ModularSet): ModularSet {
    if (this.coeff === 1 || other.coeff === 1) return EVERYTHING;
    return new ModularSet(gcd(this.coeff, other.coeff), this.base + other.base);
  }

  negate(): ModularSet {
    return new ModularSet(this.coeff, -this.base);
  }

  subtract(other: ModularSet): ModularSet {
    return this.add(other.negate());
  }

  multiply(other: ModularSet): ModularSet {
    if (this.coeff === 1 && other.coeff === 1) return EVERYTHING;
    const coeff = gcd(gcd(this.base * other.coeff, other.base * this.coeff), this.coeff * other.coeff);
    return new ModularSet(coeff, this.base * other.base);
  }

  floorDivideBy(divisor: number): ModularSet {
    if (!Number.isInteger(divisor) || divisor <= 0) return EVERYTHING;
    if (!this.divisibleBy(divisor)) return EVERYTHING;
    return this.isExact
      ? ModularSet.exact(this.base / divisor)
      : new ModularSet(this.coeff / divisor, this.base / divisor);
  }

  moduloBy(divisor: number): ModularSet {
    if (!Number.isInteger(divisor) || divisor <= 0) return EVERYTHING;
    if (this.isExact) return ModularSet.exact(floorMod(this.base, divisor));
    if (this.coeff % divisor === 0) return ModularSet.exact(floorMod(this.base, divisor));
    return new ModularSet(gcd(this.coeff, divisor), this.base);
  }

  intersectBound(min: number, max: number): { min: number; max: number } {
    if (this.isExact) return { min: Math.max(min, this.base), max: Math.min(max, this.base) };
    if (this.coeff <= 1) return { min, max };
    const low = Number.isFinite(min) ? min + floorMod(this.base - min, this.coeff) : min;
    const high = Number.isFinite(max) ? max - floorMod(max - this.base, this.coeff) : max;
    return { min: low, max: high };
  }
}

const EVERYTHING = new ModularSet(1, 0);

export function modularSetOf(expr: SymExpr, lookup: ModularLookup): ModularSet {
  if (typeof expr === 'number') return Number.isInteger(expr) ? ModularSet.exact(expr) : EVERYTHING;
  if (!(expr instanceof SymInt)) return EVERYTHING;

  if (expr.type === 'var') return lookup(expr.name as string) || EVERYTHING;

  const a = modularSetOf(expr.args[0], lookup);
  if (expr.type === 'neg') return a.negate();
  if (a.coeff === 1 && (expr.type === 'add' || expr.type === 'sub')) return EVERYTHING;

  const b = modularSetOf(expr.args[1], lookup);
  switch (expr.type) {
    case 'add': return a.add(b);
    case 'sub': return a.subtract(b);
    case 'mul': return a.multiply(b);
    case 'div': return b.isExact ? a.floorDivideBy(b.base) : EVERYTHING;
    case 'ceildiv': return b.isExact && a.divisibleBy(b.base) ? a.floorDivideBy(b.base) : EVERYTHING;
    case 'mod': return b.isExact ? a.moduloBy(b.base) : EVERYTHING;
    default: return a.coeff === b.coeff && a.base === b.base ? a : EVERYTHING;
  }
}
