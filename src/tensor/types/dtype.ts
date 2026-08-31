import {
  ScalarType as _ScalarType,
  scalarBytes as _scalarBytes,
  isFloatType as _isFloatType,
  isIntType as _isIntType,
  isNumericType as _isNumericType,
  isBoolType as _isBoolType,
  promoteDtype as _promoteDtype,
  isValuePreservingCast as _isValuePreservingCast,
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
export const canCast = _isValuePreservingCast;

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

export { resultDtype } from '../../compiler/ir/graph/types.js';

export function dtypeSize(dtype: DType): number {
  return scalarBytes(dtype);
}
