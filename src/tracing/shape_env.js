import { DYNAMIC } from '../compiler/ir/graph/types.js';
import { SymInt } from '../compiler/analysis/sym_int.js';

const _GUARD_OPS = {
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
  gt: (a, b) => a > b,
  ge: (a, b) => a >= b,
  lt: (a, b) => a < b,
  le: (a, b) => a <= b,
};

export class ShapeEnv {
  constructor() {
    this._symbols = new Map();
    this._guards = [];
    this._bindings = new Map();
    this._nextId = 0;
  }

  allocate(inputIdx, dimIdx, hint) {
    const name = `s${this._nextId++}`;
    this._symbols.set(name, { hint, inputIdx, dimIdx });
    return name;
  }

  produceShapeSpec(inputIdx, concreteShape, dynamicDims) {
    const irShape = new Array(concreteShape.length);
    const symShape = new Array(concreteShape.length);

    for (let i = 0; i < concreteShape.length; i++) {
      if (dynamicDims && dynamicDims.has(i)) {
        const sym = this.allocate(inputIdx, i, concreteShape[i]);
        irShape[i] = DYNAMIC;
        symShape[i] = sym;
      } else {
        const sym = this.allocate(inputIdx, i, concreteShape[i]);
        this.guardRelation(sym, 'eq', concreteShape[i]);
        irShape[i] = concreteShape[i];
        symShape[i] = concreteShape[i];
      }
    }

    return { irShape, symShape };
  }

  guardRelation(lhs, op, rhs) {
    this._guards.push({ lhs, op, rhs });
  }

  guardDivisible(sym, divisor) {
    this._guards.push({ type: 'divisible', sym, divisor });
  }

  bindInputShapes(inputs) {
    this._bindings.clear();
    for (const [name, info] of this._symbols) {
      this._bindings.set(name, inputs[info.inputIdx].shape[info.dimIdx]);
    }
  }

  evaluateGuards() {
    for (let i = 0; i < this._guards.length; i++) {
      const g = this._guards[i];

      if (g.type === 'divisible') {
        const val = this._resolve(g.sym);
        if (val % g.divisor !== 0) return { passed: false, failedGuard: g };
        continue;
      }

      const lVal = this._resolve(g.lhs);
      const rVal = this._resolve(g.rhs);
      if (!_GUARD_OPS[g.op](lVal, rVal)) return { passed: false, failedGuard: g };
    }

    return { passed: true, failedGuard: null };
  }

  resolveSymbolicShape(symShape) {
    const resolved = new Array(symShape.length);
    for (let i = 0; i < symShape.length; i++) {
      resolved[i] = this._resolve(symShape[i]);
    }
    return resolved;
  }

  _resolve(expr) {
    if (typeof expr === 'number') return expr;
    if (typeof expr === 'string') return this._bindings.get(expr);
    if (expr instanceof SymInt) return SymInt.evaluate(expr, this._bindings);
    return expr;
  }

  get symbols() { return this._symbols; }
  get guards() { return this._guards; }
  get bindings() { return this._bindings; }
}
