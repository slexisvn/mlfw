import { DYNAMIC } from '../compiler/ir/graph/types.js';
import { SymInt } from '../compiler/analysis/sym_int.js';
import type { TensorInput } from './types.js';
import type { SymbolicDim, MutableSymbolicShape, SymbolicShape } from './types.js';

type GuardOp = 'eq' | 'ne' | 'gt' | 'ge' | 'lt' | 'le';
type SymbolInfo = { hint: number; inputIdx: number; dimIdx: number };
type RelationGuard = { lhs: SymbolicDim; op: GuardOp; rhs: SymbolicDim };
type DivisibleGuard = { type: 'divisible'; sym: SymbolicDim; divisor: number };
type ShapeGuard = RelationGuard | DivisibleGuard;

const _GUARD_OPS: Record<GuardOp, (a: number, b: number) => boolean> = {
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
  gt: (a, b) => a > b,
  ge: (a, b) => a >= b,
  lt: (a, b) => a < b,
  le: (a, b) => a <= b,
};

export class ShapeEnv {
  private _symbols: Map<string, SymbolInfo>;
  private _guards: ShapeGuard[];
  private _bindings: Map<string, number>;
  private _nextId: number;

  constructor() {
    this._symbols = new Map();
    this._guards = [];
    this._bindings = new Map();
    this._nextId = 0;
  }

  allocate(inputIdx: number, dimIdx: number, hint: number): string {
    const name = `s${this._nextId++}`;
    this._symbols.set(name, { hint, inputIdx, dimIdx });
    return name;
  }

  produceShapeSpec(inputIdx: number, concreteShape: readonly number[], dynamicDims?: Set<number> | null): { irShape: number[]; symShape: MutableSymbolicShape } {
    const irShape = new Array<number>(concreteShape.length);
    const symShape = new Array<SymbolicDim>(concreteShape.length);

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

  guardRelation(lhs: SymbolicDim, op: GuardOp, rhs: SymbolicDim): void {
    this._guards.push({ lhs, op, rhs });
  }

  guardDivisible(sym: SymbolicDim, divisor: number): void {
    this._guards.push({ type: 'divisible', sym, divisor });
  }

  bindInputShapes(inputs: readonly TensorInput[]): void {
    this._bindings.clear();
    for (const [name, info] of this._symbols) {
      this._bindings.set(name, inputs[info.inputIdx].shape[info.dimIdx]);
    }
  }

  evaluateGuards(): { passed: true; failedGuard: null } | { passed: false; failedGuard: ShapeGuard } {
    for (let i = 0; i < this._guards.length; i++) {
      const g = this._guards[i];

      if ('type' in g) {
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

  resolveSymbolicShape(symShape: SymbolicShape): number[] {
    const resolved = new Array<number>(symShape.length);
    for (let i = 0; i < symShape.length; i++) {
      resolved[i] = this._resolve(symShape[i]);
    }
    return resolved;
  }

  _resolve(expr: SymbolicDim): number {
    if (typeof expr === 'number') return expr;
    if (typeof expr === 'string') return this._bindings.get(expr)!;
    if (expr instanceof SymInt) return SymInt.evaluate(expr, this._bindings);
    return expr;
  }

  get symbols(): Map<string, SymbolInfo> { return this._symbols; }
  get guards(): ShapeGuard[] { return this._guards; }
  get bindings(): Map<string, number> { return this._bindings; }
}
