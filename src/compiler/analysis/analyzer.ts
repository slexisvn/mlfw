import { SymInt } from './sym_int.js';
import type { SymExpr } from './sym_int.js';

const NEG_INF = -Infinity;
const POS_INF = Infinity;

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

export class Analyzer {
  private _varBounds: Map<string, IntBound>;

  constructor() {
    this._varBounds = new Map();
  }

  bind(name: string, min: number, max: number): this {
    this._varBounds.set(name, new IntBound(min, max));
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

  constIntBound(expr: SymExpr): IntBound {
    if (typeof expr === 'number') return new IntBound(expr, expr);
    if (!(expr instanceof SymInt)) return EVERYTHING;

    if (expr.type === 'var') {
      return this._varBounds.get(expr.name as string) || EVERYTHING;
    }

    const a = (expr.args.length > 0 ? this.constIntBound(expr.args[0]) : null) as IntBound;
    const b = (expr.args.length > 1 ? this.constIntBound(expr.args[1]) : null) as IntBound;

    switch (expr.type) {
      case 'add':
        return new IntBound(boundLo(a.min + b.min), boundHi(a.max + b.max));
      case 'sub':
        return new IntBound(boundLo(a.min - b.max), boundHi(a.max - b.min));
      case 'neg':
        return new IntBound(-a.max, -a.min);
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
    const bound = this.constIntBound(SymInt.sub(expr, value));
    return bound.min >= 0;
  }

  canProveLess(expr: SymExpr, value: SymExpr): boolean {
    const bound = this.constIntBound(SymInt.sub(expr, value));
    return bound.max < 0;
  }

  canProveNonNegative(expr: SymExpr): boolean {
    return this.canProveGreaterEqual(expr, 0);
  }

  canProveEqual(a: SymExpr, b: SymExpr): boolean {
    if (SymInt.equals(a, b)) return true;
    const bound = this.constIntBound(SymInt.sub(a, b));
    return bound.min === 0 && bound.max === 0;
  }
}
