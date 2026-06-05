import { DYNAMIC } from '../compiler/ir/graph/types.js';

export class ShapeEnv {
  constructor() {
    this._symbols = new Map();
    this._guards = [];
    this._nextId = 0;
  }

  createSymbolicSize(hint) {
    if (hint !== undefined && hint !== null && hint >= 0) return hint;
    const name = `s${this._nextId++}`;
    this._symbols.set(name, DYNAMIC);
    return DYNAMIC;
  }

  guard(constraint) {
    this._guards.push(constraint);
  }

  guardEquals(symName, value) {
    this._guards.push({ type: 'eq', sym: symName, value });
  }

  guardGreater(symName, value) {
    this._guards.push({ type: 'gt', sym: symName, value });
  }

  resolveShape(symbolicShape) {
    return symbolicShape.map(d => {
      if (typeof d === 'string' && this._symbols.has(d)) {
        return this._symbols.get(d);
      }
      return d;
    });
  }

  bindConcrete(symName, value) {
    this._symbols.set(symName, value);
  }

  get symbols() {
    return this._symbols;
  }

  get guards() {
    return this._guards;
  }
}
