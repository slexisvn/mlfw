import { ScalarType, scalarBytes } from './types.js';

export const QuantizationScheme = Object.freeze({
  PER_TENSOR_SYMMETRIC: 'per_tensor_symmetric',
  PER_TENSOR_ASYMMETRIC: 'per_tensor_asymmetric',
  PER_CHANNEL: 'per_channel',
  PER_GROUP: 'per_group'
});

const SCHEME_SET = new Set(Object.values(QuantizationScheme));
const QUANTIZABLE_DTYPES = new Set([ScalarType.I8, ScalarType.UI8]);

export class QuantizationParams {
  constructor(config) {
    this.scheme = config.scheme;
    this.scale = config.scale;
    this.zeroPoint = config.zeroPoint;
    this.axis = config.axis ?? null;
    this.groupSize = config.groupSize ?? null;
    this.dtype = config.dtype || ScalarType.I8;
    this.numBits = config.numBits || (scalarBytes(this.dtype) * 8);
    this._hash = null;
  }

  clampRange() {
    if (this.isSymmetric()) {
      const bound = (1 << (this.numBits - 1)) - 1;
      return [-bound, bound];
    }
    if (this.dtype === ScalarType.UI8) {
      return [0, (1 << this.numBits) - 1];
    }
    const min = -(1 << (this.numBits - 1));
    const max = (1 << (this.numBits - 1)) - 1;
    return [min, max];
  }

  quantize(floatVal) {
    const [cMin, cMax] = this.clampRange();
    const raw = Math.round(floatVal / this.getScalarScale() + this.getScalarZeroPoint());
    return Math.max(cMin, Math.min(cMax, raw));
  }

  dequantize(intVal) {
    return (intVal - this.getScalarZeroPoint()) * this.getScalarScale();
  }

  quantizeArray(floatArr) {
    const [cMin, cMax] = this.clampRange();
    const result = new Array(floatArr.length);
    if (this.isPerChannel()) {
      throw new Error('Use quantizeArrayPerChannel(floatArr, shape) for per-channel quantization');
    }
    const s = this.getScalarScale();
    const zp = this.getScalarZeroPoint();
    for (let i = 0; i < floatArr.length; i++) {
      result[i] = Math.max(cMin, Math.min(cMax, Math.round(floatArr[i] / s + zp)));
    }
    return result;
  }

  dequantizeArray(intArr) {
    const result = new Array(intArr.length);
    if (this.isPerChannel()) {
      throw new Error('Use dequantizeArrayPerChannel(intArr, shape) for per-channel dequantization');
    }
    const s = this.getScalarScale();
    const zp = this.getScalarZeroPoint();
    for (let i = 0; i < intArr.length; i++) {
      result[i] = (intArr[i] - zp) * s;
    }
    return result;
  }

  _channelStride(shape) {
    let stride = 1;
    for (let d = this.axis + 1; d < shape.length; d++) stride *= shape[d];
    return stride;
  }

  quantizeArrayPerChannel(floatArr, shape) {
    const [cMin, cMax] = this.clampRange();
    const stride = this._channelStride(shape);
    const numCh = shape[this.axis];
    const result = new Array(floatArr.length);
    for (let i = 0; i < floatArr.length; i++) {
      const ch = Math.floor(i / stride) % numCh;
      const s = this.getScaleForChannel(ch);
      const zp = this.getZeroPointForChannel(ch);
      result[i] = Math.max(cMin, Math.min(cMax, Math.round(floatArr[i] / s + zp)));
    }
    return result;
  }

  dequantizeArrayPerChannel(intArr, shape) {
    const stride = this._channelStride(shape);
    const numCh = shape[this.axis];
    const result = new Array(intArr.length);
    for (let i = 0; i < intArr.length; i++) {
      const ch = Math.floor(i / stride) % numCh;
      result[i] = (intArr[i] - this.getZeroPointForChannel(ch)) * this.getScaleForChannel(ch);
    }
    return result;
  }

  quantizeArrayPerGroup(floatArr) {
    const [cMin, cMax] = this.clampRange();
    const result = new Array(floatArr.length);
    for (let i = 0; i < floatArr.length; i++) {
      const g = Math.floor(i / this.groupSize);
      const s = this.scale[g];
      const zp = this.zeroPoint[g];
      result[i] = Math.max(cMin, Math.min(cMax, Math.round(floatArr[i] / s + zp)));
    }
    return result;
  }

  dequantizeArrayPerGroup(intArr) {
    const result = new Array(intArr.length);
    for (let i = 0; i < intArr.length; i++) {
      const g = Math.floor(i / this.groupSize);
      result[i] = (intArr[i] - this.zeroPoint[g]) * this.scale[g];
    }
    return result;
  }

  getScaleForGroup(g) {
    return this.scale[g];
  }

  getZeroPointForGroup(g) {
    return this.zeroPoint[g];
  }

  getScaleForChannel(ch) {
    if (!this.isPerChannel()) return this.getScalarScale();
    return this.scale[ch];
  }

  getZeroPointForChannel(ch) {
    if (!this.isPerChannel()) return this.getScalarZeroPoint();
    return this.zeroPoint[ch];
  }

  getScalarScale() {
    return typeof this.scale === 'number' ? this.scale : this.scale[0];
  }

  getScalarZeroPoint() {
    return typeof this.zeroPoint === 'number' ? this.zeroPoint : this.zeroPoint[0];
  }

  numChannels() {
    if (!this.isPerChannel()) return 1;
    return typeof this.scale === 'number' ? 1 : this.scale.length;
  }

  isPerChannel() {
    return this.scheme === QuantizationScheme.PER_CHANNEL;
  }

  isPerGroup() {
    return this.scheme === QuantizationScheme.PER_GROUP;
  }

  isSymmetric() {
    return this.scheme === QuantizationScheme.PER_TENSOR_SYMMETRIC;
  }

  equals(other) {
    if (this === other) return true;
    if (!(other instanceof QuantizationParams)) return false;
    if (this.scheme !== other.scheme) return false;
    if (this.dtype !== other.dtype) return false;
    if (this.numBits !== other.numBits) return false;
    if (this.axis !== other.axis) return false;
    if (this.groupSize !== other.groupSize) return false;
    if (!scaleEquals(this.scale, other.scale)) return false;
    if (!scaleEquals(this.zeroPoint, other.zeroPoint)) return false;
    return true;
  }

  hash() {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    h = ((h ^ hashStr(this.scheme)) * 0x01000193) & 0x7fffffff;
    h = ((h ^ hashStr(this.dtype)) * 0x01000193) & 0x7fffffff;
    h = ((h ^ this.numBits) * 0x01000193) & 0x7fffffff;
    h = ((h ^ hashScaleValue(this.scale)) * 0x01000193) & 0x7fffffff;
    this._hash = h;
    return h;
  }

  serialize() {
    const arrayScale = this.isPerChannel() || this.isPerGroup();
    return {
      scheme: this.scheme,
      scale: arrayScale ? [...this.scale] : this.scale,
      zeroPoint: arrayScale ? [...this.zeroPoint] : this.zeroPoint,
      axis: this.axis,
      groupSize: this.groupSize,
      dtype: this.dtype,
      numBits: this.numBits
    };
  }

  static deserialize(obj) {
    return new QuantizationParams(obj);
  }

  static fromRange(min, max, scheme, dtype = ScalarType.I8, numBits = 8) {
    if (scheme === QuantizationScheme.PER_TENSOR_SYMMETRIC) {
      const absMax = Math.max(Math.abs(min), Math.abs(max));
      const bound = (1 << (numBits - 1)) - 1;
      const scale = absMax / bound || 1e-10;
      return new QuantizationParams({ scheme, scale, zeroPoint: 0, dtype, numBits });
    }

    const [cMin, cMax] = dtype === ScalarType.UI8
      ? [0, (1 << numBits) - 1]
      : [-(1 << (numBits - 1)), (1 << (numBits - 1)) - 1];
    const range = max - min || 1e-10;
    const scale = range / (cMax - cMin);
    const zeroPoint = Math.round(cMin - min / scale);
    const clampedZP = Math.max(cMin, Math.min(cMax, zeroPoint));
    return new QuantizationParams({ scheme, scale, zeroPoint: clampedZP, dtype, numBits });
  }

  static fromRangePerChannel(mins, maxs, axis, dtype = ScalarType.I8, numBits = 8) {
    const numCh = mins.length;
    const scales = new Float64Array(numCh);
    const zeroPoints = new Int32Array(numCh);
    const bound = (1 << (numBits - 1)) - 1;
    for (let c = 0; c < numCh; c++) {
      const absMax = Math.max(Math.abs(mins[c]), Math.abs(maxs[c]));
      scales[c] = absMax / bound || 1e-10;
      zeroPoints[c] = 0;
    }
    return new QuantizationParams({
      scheme: QuantizationScheme.PER_CHANNEL,
      scale: scales, zeroPoint: zeroPoints,
      axis, dtype, numBits
    });
  }

  static defaultForActivation(scheme, dtype = ScalarType.I8, numBits = 8) {
    return QuantizationParams.fromRange(-6, 6, scheme, dtype, numBits);
  }

  static fromConstantArray(data, scheme, dtype = ScalarType.I8, numBits = 8) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    if (!isFinite(min)) min = -1;
    if (!isFinite(max)) max = 1;
    if (min === max) { min -= 0.5; max += 0.5; }
    return QuantizationParams.fromRange(min, max, scheme, dtype, numBits);
  }

  static fromConstantArrayPerChannel(data, shape, axis, dtype = ScalarType.I8, numBits = 8) {
    const numCh = shape[axis];
    let stride = 1;
    for (let d = axis + 1; d < shape.length; d++) stride *= shape[d];
    const mins = new Array(numCh).fill(Infinity);
    const maxs = new Array(numCh).fill(-Infinity);
    for (let i = 0; i < data.length; i++) {
      const ch = Math.floor(i / stride) % numCh;
      const v = data[i];
      if (v < mins[ch]) mins[ch] = v;
      if (v > maxs[ch]) maxs[ch] = v;
    }
    for (let c = 0; c < numCh; c++) {
      if (!isFinite(mins[c])) mins[c] = -1;
      if (!isFinite(maxs[c])) maxs[c] = 1;
      if (mins[c] === maxs[c]) { mins[c] -= 0.5; maxs[c] += 0.5; }
    }
    return QuantizationParams.fromRangePerChannel(mins, maxs, axis, dtype, numBits);
  }

  static fromConstantArrayPerGroup(data, groupSize, dtype = ScalarType.I8, numBits = 4) {
    const numGroups = Math.ceil(data.length / groupSize);
    const scales = new Float64Array(numGroups);
    const zeroPoints = new Int32Array(numGroups);
    const bound = (1 << (numBits - 1)) - 1;
    for (let g = 0; g < numGroups; g++) {
      const start = g * groupSize;
      const end = Math.min(start + groupSize, data.length);
      let absMax = 0;
      for (let i = start; i < end; i++) {
        const a = Math.abs(data[i]);
        if (a > absMax) absMax = a;
      }
      scales[g] = absMax / bound || 1e-10;
      zeroPoints[g] = 0;
    }
    return new QuantizationParams({
      scheme: QuantizationScheme.PER_GROUP,
      scale: scales, zeroPoint: zeroPoints,
      groupSize, dtype, numBits
    });
  }

  static isQuantizableDtype(dtype) {
    return QUANTIZABLE_DTYPES.has(dtype);
  }

  static isValidScheme(scheme) {
    return SCHEME_SET.has(scheme);
  }
}

function scaleEquals(a, b) {
  if (a === b) return true;
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  if (typeof a !== typeof b) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) & 0x7fffffff;
  }
  return h;
}

function hashScaleValue(v) {
  if (typeof v === 'number') {
    const buf = new Float64Array([v]);
    const view = new Uint32Array(buf.buffer);
    return (view[0] ^ view[1]) & 0x7fffffff;
  }
  let h = v.length;
  for (let i = 0; i < Math.min(v.length, 8); i++) {
    const buf = new Float64Array([v[i]]);
    const view = new Uint32Array(buf.buffer);
    h = ((h ^ (view[0] ^ view[1])) * 0x01000193) & 0x7fffffff;
  }
  return h;
}
