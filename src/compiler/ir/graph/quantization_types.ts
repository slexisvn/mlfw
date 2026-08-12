import { ScalarType, scalarBytes } from './types.js';
import type { ScalarDType } from './types.js';

export const QuantizationScheme = Object.freeze({
  PER_TENSOR_SYMMETRIC: 'per_tensor_symmetric',
  PER_TENSOR_ASYMMETRIC: 'per_tensor_asymmetric',
  PER_CHANNEL: 'per_channel',
  PER_GROUP: 'per_group'
});

export type QuantizationSchemeValue = (typeof QuantizationScheme)[keyof typeof QuantizationScheme];
export type ScaleValue = number | Float64Array | readonly number[];
export type ZeroPointValue = number | Int32Array | readonly number[];

export type QuantizationParamsConfig = Readonly<{
  scheme: QuantizationSchemeValue;
  scale: ScaleValue;
  zeroPoint: ZeroPointValue;
  axis?: number | null;
  groupSize?: number | null;
  dtype?: ScalarDType;
  numBits?: number;
}>;

export type SerializedQuantizationParams = {
  scheme: QuantizationSchemeValue;
  scale: ScaleValue;
  zeroPoint: ZeroPointValue;
  axis: number | null;
  groupSize: number | null;
  dtype: ScalarDType;
  numBits: number;
};

const SCHEME_SET = new Set<string>(Object.values(QuantizationScheme));
const QUANTIZABLE_DTYPES = new Set<ScalarDType>([ScalarType.I8, ScalarType.UI8]);

export class QuantizationParams {
  scheme: QuantizationSchemeValue;
  scale: ScaleValue;
  zeroPoint: ZeroPointValue;
  axis: number | null;
  groupSize: number | null;
  dtype: ScalarDType;
  numBits: number;
  private _hash: number | null;

  constructor(config: QuantizationParamsConfig) {
    this.scheme = config.scheme;
    this.scale = config.scale;
    this.zeroPoint = config.zeroPoint;
    this.axis = config.axis ?? null;
    this.groupSize = config.groupSize ?? null;
    this.dtype = config.dtype || ScalarType.I8;
    this.numBits = config.numBits || (scalarBytes(this.dtype) * 8);
    this._hash = null;
  }

  clampRange(): [number, number] {
    if (this.isSymmetric()) {
      const bound = 2 ** (this.numBits - 1) - 1;
      return [-bound, bound];
    }
    if (this.dtype === ScalarType.UI8) {
      return [0, 2 ** this.numBits - 1];
    }
    const min = -(2 ** (this.numBits - 1));
    const max = 2 ** (this.numBits - 1) - 1;
    return [min, max];
  }

  quantize(floatVal: number): number {
    const [cMin, cMax] = this.clampRange();
    const raw = Math.round(floatVal / this.getScalarScale() + this.getScalarZeroPoint());
    return Math.max(cMin, Math.min(cMax, raw));
  }

  dequantize(intVal: number): number {
    return (intVal - this.getScalarZeroPoint()) * this.getScalarScale();
  }

  quantizeArray(floatArr: ArrayLike<number>): number[] {
    const [cMin, cMax] = this.clampRange();
    const result = new Array<number>(floatArr.length);
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

  dequantizeArray(intArr: ArrayLike<number>): number[] {
    const result = new Array<number>(intArr.length);
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

  _channelStride(shape: readonly number[]): number {
    let stride = 1;
    for (let d = (this.axis as number) + 1; d < shape.length; d++) stride *= shape[d];
    return stride;
  }

  quantizeArrayPerChannel(floatArr: ArrayLike<number>, shape: readonly number[]): number[] {
    const [cMin, cMax] = this.clampRange();
    const stride = this._channelStride(shape);
    const numCh = shape[this.axis as number];
    const result = new Array<number>(floatArr.length);
    for (let i = 0; i < floatArr.length; i++) {
      const ch = Math.floor(i / stride) % numCh;
      const s = this.getScaleForChannel(ch);
      const zp = this.getZeroPointForChannel(ch);
      result[i] = Math.max(cMin, Math.min(cMax, Math.round(floatArr[i] / s + zp)));
    }
    return result;
  }

  dequantizeArrayPerChannel(intArr: ArrayLike<number>, shape: readonly number[]): number[] {
    const stride = this._channelStride(shape);
    const numCh = shape[this.axis as number];
    const result = new Array<number>(intArr.length);
    for (let i = 0; i < intArr.length; i++) {
      const ch = Math.floor(i / stride) % numCh;
      result[i] = (intArr[i] - this.getZeroPointForChannel(ch)) * this.getScaleForChannel(ch);
    }
    return result;
  }

  quantizeArrayPerGroup(floatArr: ArrayLike<number>): number[] {
    const [cMin, cMax] = this.clampRange();
    const result = new Array<number>(floatArr.length);
    for (let i = 0; i < floatArr.length; i++) {
      const g = Math.floor(i / (this.groupSize as number));
      const s = (this.scale as ArrayLike<number>)[g];
      const zp = (this.zeroPoint as ArrayLike<number>)[g];
      result[i] = Math.max(cMin, Math.min(cMax, Math.round(floatArr[i] / s + zp)));
    }
    return result;
  }

  dequantizeArrayPerGroup(intArr: ArrayLike<number>): number[] {
    const result = new Array<number>(intArr.length);
    for (let i = 0; i < intArr.length; i++) {
      const g = Math.floor(i / (this.groupSize as number));
      result[i] = (intArr[i] - (this.zeroPoint as ArrayLike<number>)[g]) * (this.scale as ArrayLike<number>)[g];
    }
    return result;
  }

  getScaleForGroup(g: number): number {
    return (this.scale as ArrayLike<number>)[g];
  }

  getZeroPointForGroup(g: number): number {
    return (this.zeroPoint as ArrayLike<number>)[g];
  }

  getScaleForChannel(ch: number): number {
    if (!this.isPerChannel()) return this.getScalarScale();
    return (this.scale as ArrayLike<number>)[ch];
  }

  getZeroPointForChannel(ch: number): number {
    if (!this.isPerChannel()) return this.getScalarZeroPoint();
    return (this.zeroPoint as ArrayLike<number>)[ch];
  }

  getScalarScale(): number {
    return typeof this.scale === 'number' ? this.scale : (this.scale as ArrayLike<number>)[0];
  }

  getScalarZeroPoint(): number {
    return typeof this.zeroPoint === 'number' ? this.zeroPoint : (this.zeroPoint as ArrayLike<number>)[0];
  }

  numChannels(): number {
    if (!this.isPerChannel()) return 1;
    return typeof this.scale === 'number' ? 1 : this.scale.length;
  }

  isPerChannel(): boolean {
    return this.scheme === QuantizationScheme.PER_CHANNEL;
  }

  isPerGroup(): boolean {
    return this.scheme === QuantizationScheme.PER_GROUP;
  }

  isSymmetric(): boolean {
    return this.scheme === QuantizationScheme.PER_TENSOR_SYMMETRIC;
  }

  equals(other: unknown): boolean {
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

  hash(): number {
    if (this._hash !== null) return this._hash;
    let h = 0x811c9dc5;
    h = ((h ^ hashStr(this.scheme)) * 0x01000193) & 0x7fffffff;
    h = ((h ^ hashStr(this.dtype)) * 0x01000193) & 0x7fffffff;
    h = ((h ^ this.numBits) * 0x01000193) & 0x7fffffff;
    h = ((h ^ hashScaleValue(this.scale)) * 0x01000193) & 0x7fffffff;
    this._hash = h;
    return h;
  }

  serialize(): SerializedQuantizationParams {
    const arrayScale = this.isPerChannel() || this.isPerGroup();
    return {
      scheme: this.scheme,
      scale: arrayScale ? [...(this.scale as ArrayLike<number> & Iterable<number>)] : this.scale,
      zeroPoint: arrayScale ? [...(this.zeroPoint as ArrayLike<number> & Iterable<number>)] : this.zeroPoint,
      axis: this.axis,
      groupSize: this.groupSize,
      dtype: this.dtype,
      numBits: this.numBits
    };
  }

  static deserialize(obj: QuantizationParamsConfig): QuantizationParams {
    return new QuantizationParams(obj);
  }

  static fromRange(min: number, max: number, scheme: QuantizationSchemeValue, dtype: ScalarDType = ScalarType.I8, numBits = 8): QuantizationParams {
    if (scheme === QuantizationScheme.PER_TENSOR_SYMMETRIC) {
      const absMax = Math.max(Math.abs(min), Math.abs(max));
      const bound = 2 ** (numBits - 1) - 1;
      const scale = absMax / bound || 1e-10;
      return new QuantizationParams({ scheme, scale, zeroPoint: 0, dtype, numBits });
    }

    const [cMin, cMax] = dtype === ScalarType.UI8
      ? [0, 2 ** numBits - 1]
      : [-(2 ** (numBits - 1)), 2 ** (numBits - 1) - 1];
    const range = max - min || 1e-10;
    const scale = range / (cMax - cMin);
    const zeroPoint = Math.round(cMin - min / scale);
    const clampedZP = Math.max(cMin, Math.min(cMax, zeroPoint));
    return new QuantizationParams({ scheme, scale, zeroPoint: clampedZP, dtype, numBits });
  }

  static fromRangePerChannel(mins: ArrayLike<number>, maxs: ArrayLike<number>, axis: number, dtype: ScalarDType = ScalarType.I8, numBits = 8): QuantizationParams {
    const numCh = mins.length;
    const scales = new Float64Array(numCh);
    const zeroPoints = new Int32Array(numCh);
    const bound = 2 ** (numBits - 1) - 1;
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

  static defaultForActivation(scheme: QuantizationSchemeValue, dtype: ScalarDType = ScalarType.I8, numBits = 8): QuantizationParams {
    return QuantizationParams.fromRange(-6, 6, scheme, dtype, numBits);
  }

  static fromConstantArray(data: ArrayLike<number>, scheme: QuantizationSchemeValue, dtype: ScalarDType = ScalarType.I8, numBits = 8): QuantizationParams {
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

  static fromConstantArrayPerChannel(data: ArrayLike<number>, shape: readonly number[], axis: number, dtype: ScalarDType = ScalarType.I8, numBits = 8): QuantizationParams {
    const numCh = shape[axis];
    let stride = 1;
    for (let d = axis + 1; d < shape.length; d++) stride *= shape[d];
    const mins = new Array<number>(numCh).fill(Infinity);
    const maxs = new Array<number>(numCh).fill(-Infinity);
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

  static fromConstantArrayPerGroup(data: ArrayLike<number>, groupSize: number, dtype: ScalarDType = ScalarType.I8, numBits = 4): QuantizationParams {
    const numGroups = Math.ceil(data.length / groupSize);
    const scales = new Float64Array(numGroups);
    const zeroPoints = new Int32Array(numGroups);
    const bound = 2 ** (numBits - 1) - 1;
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

  static isQuantizableDtype(dtype: ScalarDType): boolean {
    return QUANTIZABLE_DTYPES.has(dtype);
  }

  static isValidScheme(scheme: string): boolean {
    return SCHEME_SET.has(scheme);
  }
}

function scaleEquals(a: ScaleValue | ZeroPointValue, b: ScaleValue | ZeroPointValue): boolean {
  if (a === b) return true;
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  if (typeof a !== typeof b) return false;
  const av = a as ArrayLike<number>;
  const bv = b as ArrayLike<number>;
  if (av.length !== bv.length) return false;
  for (let i = 0; i < av.length; i++) {
    if (av[i] !== bv[i]) return false;
  }
  return true;
}

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) & 0x7fffffff;
  }
  return h;
}

function hashScaleValue(v: ScaleValue): number {
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
