import { SymInt } from '../../analysis/sym_int.js';

export type SymIntValue = InstanceType<typeof SymInt>;
export type Dim = number | SymIntValue;
export type Shape = readonly Dim[];
export type ScalarDType = `${ScalarType}`;
export type IRType = TensorType | TupleType | TokenType | FunctionType;

export interface HashableAttr {
  hash(): number;
  equals(other: unknown): boolean;
}

export type AttrValue =
  | number
  | string
  | boolean
  | null
  | ArrayBufferView
  | SymIntValue
  | HashableAttr
  | readonly AttrValue[];

export type AttrInit = ReadonlyMap<string, AttrValue> | Readonly<Record<string, AttrValue>>;

export function dimEquals(a: Dim, b: Dim): boolean {
  if (a === b) return true;
  if (a instanceof SymInt && b instanceof SymInt) return SymInt.equals(a, b);
  return false;
}

export enum ScalarType {
  F16 = 'f16',
  BF16 = 'bf16',
  F32 = 'f32',
  F64 = 'f64',
  I8 = 'i8',
  I16 = 'i16',
  I32 = 'i32',
  I64 = 'i64',
  UI8 = 'ui8',
  BOOL = 'bool',
  INDEX = 'index'
}

const SCALAR_BYTES = Object.freeze({
  [ScalarType.F16]: 2,
  [ScalarType.BF16]: 2,
  [ScalarType.F32]: 4,
  [ScalarType.F64]: 8,
  [ScalarType.I8]: 1,
  [ScalarType.I16]: 2,
  [ScalarType.I32]: 4,
  [ScalarType.I64]: 8,
  [ScalarType.UI8]: 1,
  [ScalarType.BOOL]: 1,
  [ScalarType.INDEX]: 4
});

const FLOAT_TYPES = new Set<ScalarDType>([ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64]);
const INT_TYPES = new Set<ScalarDType>([ScalarType.I8, ScalarType.I16, ScalarType.I32, ScalarType.I64, ScalarType.UI8]);

export const DYNAMIC = -1;

export function shapeProduct<T extends number>(shape: Shape, dynamicValue: T): number | T {
  let n = 1;
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i];
    if (typeof d !== 'number' || d < 0) return dynamicValue;
    n *= d;
  }
  return n;
}

export function symbolicShapeProduct(shape: Shape): number | SymIntValue {
  let prod: number | SymIntValue = 1;
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i];
    if (typeof d === 'number') {
      if (d < 0) return DYNAMIC;
      prod = SymInt.mul(prod, d);
    } else if (d instanceof SymInt) {
      prod = SymInt.mul(prod, d);
    } else {
      return DYNAMIC;
    }
  }
  return prod;
}

export function resolveInferredDims(shape: Shape, sourceShape: Shape): Dim[] {
  const out: Dim[] = [...shape];
  let inferAt = -1;
  for (let i = 0; i < out.length; i++) {
    if (out[i] !== DYNAMIC) continue;
    if (inferAt >= 0) return out;
    inferAt = i;
  }
  if (inferAt < 0) return out;

  const total = symbolicShapeProduct(sourceShape);
  if (total === DYNAMIC) return out;

  let known: number | SymIntValue = 1;
  for (let i = 0; i < out.length; i++) {
    if (i === inferAt) continue;
    const d = out[i];
    if (typeof d === 'number' ? d < 0 : !(d instanceof SymInt)) return out;
    known = SymInt.mul(known, d);
  }
  if (known === 0) {
    out[inferAt] = 0;
    return out;
  }
  if (typeof total === 'number' && typeof known === 'number') {
    if (total % known !== 0) return out;
    out[inferAt] = total / known;
    return out;
  }
  out[inferAt] = SymInt.div(total, known);
  return out;
}

export function scalarBytes(dtype: ScalarDType): number {
  const b = SCALAR_BYTES[dtype];
  if (b === undefined) throw new Error(`Unknown dtype: ${dtype}`);
  return b;
}

export function isFloatType(dtype: ScalarDType): boolean { return FLOAT_TYPES.has(dtype); }
export function isIntType(dtype: ScalarDType): boolean { return INT_TYPES.has(dtype); }
export function isNumericType(dtype: ScalarDType): boolean { return FLOAT_TYPES.has(dtype) || INT_TYPES.has(dtype); }
export function isBoolType(dtype: ScalarDType): boolean { return dtype === ScalarType.BOOL; }

const VALUE_PRESERVING_WIDENINGS: Readonly<Partial<Record<ScalarDType, readonly ScalarDType[]>>> = {
  [ScalarType.BOOL]: [ScalarType.I8, ScalarType.I16, ScalarType.I32, ScalarType.I64, ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64],
  [ScalarType.UI8]: [ScalarType.I16, ScalarType.I32, ScalarType.I64, ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64],
  [ScalarType.I8]: [ScalarType.I16, ScalarType.I32, ScalarType.I64, ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64],
  [ScalarType.I16]: [ScalarType.I32, ScalarType.I64, ScalarType.F32, ScalarType.F64],
  [ScalarType.I32]: [ScalarType.I64, ScalarType.F64],
  [ScalarType.F16]: [ScalarType.F32, ScalarType.F64],
  [ScalarType.BF16]: [ScalarType.F32, ScalarType.F64],
  [ScalarType.F32]: [ScalarType.F64]
};

const VALUE_PRESERVING_TARGETS = new Map<ScalarDType, ReadonlySet<ScalarDType>>();
for (const [from, targets] of Object.entries(VALUE_PRESERVING_WIDENINGS)) {
  VALUE_PRESERVING_TARGETS.set(from as ScalarDType, new Set(targets as readonly ScalarDType[]));
}

export function isValuePreservingCast(from: ScalarDType, to: ScalarDType): boolean {
  if (from === to) return true;
  const targets = VALUE_PRESERVING_TARGETS.get(from);
  return !!targets && targets.has(to);
}

export function promoteDtype(a: ScalarDType, b: ScalarDType): ScalarDType | null {
  if (a === b) return a;
  const ai = isIntType(a), af = isFloatType(a);
  const bi = isIntType(b), bf = isFloatType(b);
  if (af && bf) {
    return scalarBytes(a) >= scalarBytes(b) ? a : b;
  }
  if (ai && bi) {
    return scalarBytes(a) >= scalarBytes(b) ? a : b;
  }
  if (af && bi) return a;
  if (ai && bf) return b;
  return null;
}

export class Layout {
  readonly order: readonly number[];
  private _hash: number | null;

  constructor(order: readonly number[]) {
    this.order = Object.freeze([...order]);
    this._hash = null;
  }

  static rowMajor(rank: number): Layout {
    const order = new Array(rank);
    for (let i = 0; i < rank; i++) order[i] = i;
    return new Layout(order);
  }

  static columnMajor(rank: number): Layout {
    const order = new Array(rank);
    for (let i = 0; i < rank; i++) order[i] = rank - 1 - i;
    return new Layout(order);
  }

  get rank() { return this.order.length; }

  isIdentity() {
    for (let i = 0; i < this.order.length; i++) {
      if (this.order[i] !== i) return false;
    }
    return true;
  }

  inverse() {
    const inv = new Array(this.order.length);
    for (let i = 0; i < this.order.length; i++) inv[this.order[i]] = i;
    return new Layout(inv);
  }

  compose(other: Layout): Layout {
    if (this.order.length !== other.order.length) {
      throw new Error('Cannot compose layouts of different ranks');
    }
    const result = new Array<number>(this.order.length);
    for (let i = 0; i < this.order.length; i++) {
      result[i] = this.order[other.order[i]];
    }
    return new Layout(result);
  }

  computeStrides(shape: Shape): number[] {
    const n = shape.length;
    const strides = new Array(n);
    let stride = 1;
    for (let i = n - 1; i >= 0; i--) {
      const dim = this.order[i];
      strides[dim] = stride;
      if (shape[dim] === DYNAMIC || shape[dim] instanceof SymInt) {
        stride = DYNAMIC;
      } else if (stride !== DYNAMIC) {
        stride *= shape[dim] as number;
      }
    }
    return strides;
  }

  equals(other: unknown): boolean {
    if (this === other) return true;
    if (!(other instanceof Layout)) return false;
    if (this.order.length !== other.order.length) return false;
    for (let i = 0; i < this.order.length; i++) {
      if (this.order[i] !== other.order[i]) return false;
    }
    return true;
  }

  hash(): number {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    for (let i = 0; i < this.order.length; i++) {
      h = ((h ^ this.order[i]) * 0x01000193) & 0x7fffffff;
    }
    this._hash = h;
    return h;
  }
}

export function broadcastDim(a: Dim, b: Dim): Dim | null {
  if (dimEquals(a, b)) return a;
  if (a === 1) return b;
  if (b === 1) return a;
  if (a === DYNAMIC) return b === DYNAMIC ? DYNAMIC : b;
  if (b === DYNAMIC) return a;
  if (a instanceof SymInt || b instanceof SymInt) return DYNAMIC;
  return null;
}

export class TensorType {
  readonly shape: readonly Dim[];
  readonly dtype: ScalarDType;
  readonly layout: Layout;
  private _hash: number | null;

  constructor(shape: Shape, dtype: ScalarDType, layout: Layout | null = null) {
    this.shape = Object.freeze([...shape]);
    this.dtype = dtype;
    this.layout = layout || Layout.rowMajor(shape.length);
    this._hash = null;
  }

  get rank() { return this.shape.length; }
  get isScalar() { return this.shape.length === 0; }
  get isFullyStatic() { return this.shape.every(d => typeof d === 'number' && d >= 0); }
  get hasDynamic() { return this.shape.some(d => d === DYNAMIC || d instanceof SymInt); }

  numel(): number {
    return shapeProduct(this.shape, DYNAMIC);
  }

  symbolicNumel(): number | SymIntValue {
    return symbolicShapeProduct(this.shape);
  }

  sizeInBytes(): number {
    const n = this.numel();
    return n === DYNAMIC ? DYNAMIC : n * SCALAR_BYTES[this.dtype];
  }

  strides(): number[] {
    return this.layout.computeStrides(this.shape);
  }

  withShape(s: Shape): TensorType { return new TensorType(s, this.dtype, this.layout); }
  withDtype(d: ScalarDType): TensorType { return new TensorType(this.shape, d, this.layout); }
  withLayout(l: Layout): TensorType { return new TensorType(this.shape, this.dtype, l); }

  equals(other: unknown): boolean {
    if (this === other) return true;
    if (!(other instanceof TensorType)) return false;
    if (this.dtype !== other.dtype) return false;
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (!dimEquals(this.shape[i], other.shape[i])) return false;
    }
    return this.layout.equals(other.layout);
  }

  shapeEquals(other: unknown): boolean {
    if (!(other instanceof TensorType)) return false;
    if (this.dtype !== other.dtype) return false;
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (!dimEquals(this.shape[i], other.shape[i])) return false;
    }
    return true;
  }

  shapeCompatible(other: TensorType): boolean {
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      const a = this.shape[i], b = other.shape[i];
      if (a === DYNAMIC || b === DYNAMIC) continue;
      if (dimEquals(a, b)) continue;
      if (typeof a === 'number' && typeof b === 'number') return false;
    }
    return true;
  }

  hash(): number {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    for (let i = 0; i < this.shape.length; i++) {
      const dim = this.shape[i];
      const d = typeof dim === 'number' ? (dim & 0xffff) : 0x7fff;
      h = ((h ^ d) * 0x01000193) & 0x7fffffff;
    }
    h = ((h ^ this.dtype.charCodeAt(0)) * 0x01000193) & 0x7fffffff;
    this._hash = h;
    return h;
  }

  static broadcastShape(...shapes: Shape[]): Dim[] | null {
    let maxRank = 0;
    for (let i = 0; i < shapes.length; i++) {
      if (shapes[i].length > maxRank) maxRank = shapes[i].length;
    }
    const result = new Array<Dim>(maxRank);
    for (let i = 0; i < maxRank; i++) {
      let dim: Dim = 1;
      for (let j = 0; j < shapes.length; j++) {
        const s = shapes[j];
        const si = i - (maxRank - s.length);
        if (si < 0) continue;
        const next = broadcastDim(dim, s[si]);
        if (next === null) return null;
        dim = next;
      }
      result[i] = dim;
    }
    return result;
  }

  static broadcastCompatible(...shapes: Shape[]): boolean {
    return TensorType.broadcastShape(...shapes) !== null;
  }
}

export class TupleType {
  readonly elements: readonly IRType[];

  constructor(elements: readonly IRType[]) {
    this.elements = Object.freeze([...elements]);
  }

  equals(other: unknown): boolean {
    if (this === other) return true;
    if (!(other instanceof TupleType)) return false;
    if (this.elements.length !== other.elements.length) return false;
    for (let i = 0; i < this.elements.length; i++) {
      if (!typeEquals(this.elements[i], other.elements[i])) return false;
    }
    return true;
  }
}

export class TokenType {
  equals(other: unknown): boolean { return other instanceof TokenType; }
}

export class FunctionType {
  readonly inputs: readonly IRType[];
  readonly outputs: readonly IRType[];

  constructor(inputs: readonly IRType[], outputs: readonly IRType[]) {
    this.inputs = Object.freeze([...inputs]);
    this.outputs = Object.freeze([...outputs]);
  }

  equals(other: unknown): boolean {
    if (this === other) return true;
    if (!(other instanceof FunctionType)) return false;
    if (this.inputs.length !== other.inputs.length) return false;
    if (this.outputs.length !== other.outputs.length) return false;
    for (let i = 0; i < this.inputs.length; i++) {
      if (!typeEquals(this.inputs[i], other.inputs[i])) return false;
    }
    for (let i = 0; i < this.outputs.length; i++) {
      if (!typeEquals(this.outputs[i], other.outputs[i])) return false;
    }
    return true;
  }
}

export function typeEquals(a: IRType, b: IRType): boolean {
  if (a === b) return true;
  if (a && typeof a.equals === 'function') return a.equals(b);
  return false;
}

export function dimToString(dim: Dim): string {
  if (dim === DYNAMIC) return '?';
  if (typeof dim === 'number') return String(dim);
  return `[${String(dim)}]`;
}

export function typeToString(type: IRType): string {
  if (type instanceof TensorType) {
    const dims = type.shape.map(dimToString);
    const layout = type.layout.isIdentity() ? '' : `, [${type.layout.order.join(', ')}]`;
    return `tensor<${dims.join('x')}x${type.dtype}${layout}>`;
  }
  if (type instanceof TupleType) {
    return `tuple<${type.elements.map(typeToString).join(', ')}>`;
  }
  if (type instanceof TokenType) return 'token';
  if (type instanceof FunctionType) {
    const ins = type.inputs.map(typeToString).join(', ');
    const outs = type.outputs.map(typeToString).join(', ');
    return `(${ins}) -> (${outs})`;
  }
  return 'unknown';
}
