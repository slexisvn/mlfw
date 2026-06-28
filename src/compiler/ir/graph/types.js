export const ScalarType = Object.freeze({
  F16: 'f16',
  BF16: 'bf16',
  F32: 'f32',
  F64: 'f64',
  I8: 'i8',
  I16: 'i16',
  I32: 'i32',
  I64: 'i64',
  UI8: 'ui8',
  BOOL: 'bool',
  INDEX: 'index'
});

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

const FLOAT_TYPES = new Set([ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64]);
const INT_TYPES = new Set([ScalarType.I8, ScalarType.I16, ScalarType.I32, ScalarType.I64, ScalarType.UI8]);

export const DYNAMIC = -1;

export function shapeProduct(shape, dynamicValue) {
  let n = 1;
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i];
    if (typeof d !== 'number' || d < 0) return dynamicValue;
    n *= d;
  }
  return n;
}

export function scalarBytes(dtype) {
  const b = SCALAR_BYTES[dtype];
  if (b === undefined) throw new Error(`Unknown dtype: ${dtype}`);
  return b;
}

export function isFloatType(dtype) { return FLOAT_TYPES.has(dtype); }
export function isIntType(dtype) { return INT_TYPES.has(dtype); }
export function isNumericType(dtype) { return FLOAT_TYPES.has(dtype) || INT_TYPES.has(dtype); }
export function isBoolType(dtype) { return dtype === ScalarType.BOOL; }

export function promoteDtype(a, b) {
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
  constructor(order) {
    this.order = Object.freeze([...order]);
    this._hash = null;
  }

  static rowMajor(rank) {
    const order = new Array(rank);
    for (let i = 0; i < rank; i++) order[i] = i;
    return new Layout(order);
  }

  static columnMajor(rank) {
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

  compose(other) {
    if (this.order.length !== other.order.length) {
      throw new Error('Cannot compose layouts of different ranks');
    }
    const result = new Array(this.order.length);
    for (let i = 0; i < this.order.length; i++) {
      result[i] = this.order[other.order[i]];
    }
    return new Layout(result);
  }

  computeStrides(shape) {
    const n = shape.length;
    const strides = new Array(n);
    let stride = 1;
    for (let i = n - 1; i >= 0; i--) {
      const dim = this.order[i];
      strides[dim] = stride;
      if (shape[dim] === DYNAMIC) {
        stride = DYNAMIC;
      } else if (stride !== DYNAMIC) {
        stride *= shape[dim];
      }
    }
    return strides;
  }

  equals(other) {
    if (this === other) return true;
    if (!(other instanceof Layout)) return false;
    if (this.order.length !== other.order.length) return false;
    for (let i = 0; i < this.order.length; i++) {
      if (this.order[i] !== other.order[i]) return false;
    }
    return true;
  }

  hash() {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    for (let i = 0; i < this.order.length; i++) {
      h = ((h ^ this.order[i]) * 0x01000193) & 0x7fffffff;
    }
    this._hash = h;
    return h;
  }
}

export function broadcastDim(a, b) {
  if (a === b) return a;
  if (a === 1) return b;
  if (b === 1) return a;
  if (a === DYNAMIC) return b === DYNAMIC ? DYNAMIC : b;
  if (b === DYNAMIC) return a;
  return null;
}

export class TensorType {
  constructor(shape, dtype, layout = null) {
    this.shape = Object.freeze([...shape]);
    this.dtype = dtype;
    this.layout = layout || Layout.rowMajor(shape.length);
    this._hash = null;
  }

  get rank() { return this.shape.length; }
  get isScalar() { return this.shape.length === 0; }
  get isFullyStatic() { return this.shape.every(d => d >= 0); }
  get hasDynamic() { return this.shape.some(d => d === DYNAMIC); }

  numel() {
    return shapeProduct(this.shape, DYNAMIC);
  }

  sizeInBytes() {
    const n = this.numel();
    return n === DYNAMIC ? DYNAMIC : n * SCALAR_BYTES[this.dtype];
  }

  strides() {
    return this.layout.computeStrides(this.shape);
  }

  withShape(s) { return new TensorType(s, this.dtype, this.layout); }
  withDtype(d) { return new TensorType(this.shape, d, this.layout); }
  withLayout(l) { return new TensorType(this.shape, this.dtype, l); }

  equals(other) {
    if (this === other) return true;
    if (!(other instanceof TensorType)) return false;
    if (this.dtype !== other.dtype) return false;
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] !== other.shape[i]) return false;
    }
    return this.layout.equals(other.layout);
  }

  shapeEquals(other) {
    if (!(other instanceof TensorType)) return false;
    if (this.dtype !== other.dtype) return false;
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] !== other.shape[i]) return false;
    }
    return true;
  }

  shapeCompatible(other) {
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      const a = this.shape[i], b = other.shape[i];
      if (a === DYNAMIC || b === DYNAMIC) continue;
      if (a !== b) return false;
    }
    return true;
  }

  hash() {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    for (let i = 0; i < this.shape.length; i++) {
      h = ((h ^ (this.shape[i] & 0xffff)) * 0x01000193) & 0x7fffffff;
    }
    h = ((h ^ this.dtype.charCodeAt(0)) * 0x01000193) & 0x7fffffff;
    this._hash = h;
    return h;
  }

  static broadcastShape(...shapes) {
    let maxRank = 0;
    for (let i = 0; i < shapes.length; i++) {
      if (shapes[i].length > maxRank) maxRank = shapes[i].length;
    }
    const result = new Array(maxRank);
    for (let i = 0; i < maxRank; i++) {
      let dim = 1;
      for (let j = 0; j < shapes.length; j++) {
        const s = shapes[j];
        const si = i - (maxRank - s.length);
        if (si < 0) continue;
        dim = broadcastDim(dim, s[si]);
        if (dim === null) return null;
      }
      result[i] = dim;
    }
    return result;
  }

  static broadcastCompatible(...shapes) {
    return TensorType.broadcastShape(...shapes) !== null;
  }
}

export class TupleType {
  constructor(elements) {
    this.elements = Object.freeze([...elements]);
  }

  equals(other) {
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
  equals(other) { return other instanceof TokenType; }
}

export class FunctionType {
  constructor(inputs, outputs) {
    this.inputs = Object.freeze([...inputs]);
    this.outputs = Object.freeze([...outputs]);
  }

  equals(other) {
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

export function typeEquals(a, b) {
  if (a === b) return true;
  if (a && typeof a.equals === 'function') return a.equals(b);
  return false;
}

export function typeToString(type) {
  if (type instanceof TensorType) {
    const dims = type.shape.map(d => d === DYNAMIC ? '?' : String(d));
    return `tensor<${dims.join('x')}x${type.dtype}>`;
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
