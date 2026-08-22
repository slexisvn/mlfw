import {
  ScalarType as _ScalarType,
  scalarBytes as _scalarBytes,
  isFloatType as _isFloatType,
  isIntType as _isIntType,
  isNumericType as _isNumericType,
  isBoolType as _isBoolType,
  promoteDtype as _promoteDtype,
} from '../../compiler/ir/graph/types.js';
import { typedArrayCtor as dtypeTypedArrayCtor } from '../../util/dtype_map.js';
import type { ScalarDType } from '../../compiler/ir/graph/types.js';

export const ScalarType = _ScalarType;
export const scalarBytes = _scalarBytes;
export const isFloatType = _isFloatType;
export const isIntType = _isIntType;
export const isNumericType = _isNumericType;
export const isBoolType = _isBoolType;
export const promoteDtype = _promoteDtype;

export type DType = ScalarDType;
export type NumericTypedArray =
  | Uint16Array
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array;
export type NumericTypedArrayConstructor =
  | Uint16ArrayConstructor
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | Int8ArrayConstructor
  | Int16ArrayConstructor
  | Int32ArrayConstructor
  | BigInt64ArrayConstructor
  | Uint8ArrayConstructor;

export function typedArrayCtor(dtype: DType): NumericTypedArrayConstructor {
  return dtypeTypedArrayCtor(dtype);
}

const _VALID_DTYPES = new Set<string>(Object.values(ScalarType));

export function isDType(dtype: unknown): dtype is DType {
  return typeof dtype === 'string' && _VALID_DTYPES.has(dtype);
}

export function dtypeNames(): string[] {
  return [..._VALID_DTYPES];
}

const _FLOAT_PRECEDENCE = [ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64];
const _INT_PRECEDENCE = [ScalarType.UI8, ScalarType.I8, ScalarType.I16, ScalarType.I32, ScalarType.I64];

const _PRECEDENCE_MAP = new Map<DType, number>();
for (let i = 0; i < _FLOAT_PRECEDENCE.length; i++) _PRECEDENCE_MAP.set(_FLOAT_PRECEDENCE[i], 100 + i);
for (let i = 0; i < _INT_PRECEDENCE.length; i++) _PRECEDENCE_MAP.set(_INT_PRECEDENCE[i], i);
_PRECEDENCE_MAP.set(ScalarType.BOOL, -1);
_PRECEDENCE_MAP.set(ScalarType.INDEX, 50);

export function resultDtype(a: DType, b: DType): DType {
  if (a === b) return a;
  const pa = _PRECEDENCE_MAP.get(a) ?? 0;
  const pb = _PRECEDENCE_MAP.get(b) ?? 0;
  return pa >= pb ? a : b;
}

const _CAST_TABLE = new Map<DType, Set<DType>>();

function _addCastPair(from: DType, to: DType) {
  let set = _CAST_TABLE.get(from);
  if (!set) { set = new Set(); _CAST_TABLE.set(from, set); }
  set.add(to);
}

_addCastPair(ScalarType.BOOL, ScalarType.I8);
_addCastPair(ScalarType.BOOL, ScalarType.I16);
_addCastPair(ScalarType.BOOL, ScalarType.I32);
_addCastPair(ScalarType.BOOL, ScalarType.I64);
_addCastPair(ScalarType.BOOL, ScalarType.F16);
_addCastPair(ScalarType.BOOL, ScalarType.F32);
_addCastPair(ScalarType.BOOL, ScalarType.F64);
_addCastPair(ScalarType.UI8, ScalarType.I16);
_addCastPair(ScalarType.UI8, ScalarType.I32);
_addCastPair(ScalarType.UI8, ScalarType.I64);
_addCastPair(ScalarType.UI8, ScalarType.F16);
_addCastPair(ScalarType.UI8, ScalarType.F32);
_addCastPair(ScalarType.UI8, ScalarType.F64);
_addCastPair(ScalarType.I8, ScalarType.I16);
_addCastPair(ScalarType.I8, ScalarType.I32);
_addCastPair(ScalarType.I8, ScalarType.I64);
_addCastPair(ScalarType.I8, ScalarType.F16);
_addCastPair(ScalarType.I8, ScalarType.F32);
_addCastPair(ScalarType.I8, ScalarType.F64);
_addCastPair(ScalarType.I16, ScalarType.I32);
_addCastPair(ScalarType.I16, ScalarType.I64);
_addCastPair(ScalarType.I16, ScalarType.F32);
_addCastPair(ScalarType.I16, ScalarType.F64);
_addCastPair(ScalarType.I32, ScalarType.I64);
_addCastPair(ScalarType.I32, ScalarType.F64);
_addCastPair(ScalarType.F16, ScalarType.F32);
_addCastPair(ScalarType.F16, ScalarType.F64);
_addCastPair(ScalarType.BF16, ScalarType.F32);
_addCastPair(ScalarType.BF16, ScalarType.F64);
_addCastPair(ScalarType.BOOL, ScalarType.BF16);
_addCastPair(ScalarType.UI8, ScalarType.BF16);
_addCastPair(ScalarType.I8, ScalarType.BF16);
_addCastPair(ScalarType.F32, ScalarType.F64);

export function canCast(from: DType, to: DType): boolean {
  if (from === to) return true;
  const set = _CAST_TABLE.get(from);
  return set ? set.has(to) : false;
}

export function dtypeSize(dtype: DType): number {
  return scalarBytes(dtype);
}
