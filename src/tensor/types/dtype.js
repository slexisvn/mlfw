import {
  ScalarType as _ScalarType,
  scalarBytes as _scalarBytes,
  isFloatType as _isFloatType,
  isIntType as _isIntType,
  isNumericType as _isNumericType,
  isBoolType as _isBoolType,
  promoteDtype as _promoteDtype,
} from '../../compiler/ir/graph/types.js';

export const ScalarType = _ScalarType;
export const scalarBytes = _scalarBytes;
export const isFloatType = _isFloatType;
export const isIntType = _isIntType;
export const isNumericType = _isNumericType;
export const isBoolType = _isBoolType;
export const promoteDtype = _promoteDtype;

const _TYPED_ARRAY_CTORS = Object.freeze({
  [ScalarType.F16]: Uint16Array,
  [ScalarType.BF16]: Uint16Array,
  [ScalarType.F32]: Float32Array,
  [ScalarType.F64]: Float64Array,
  [ScalarType.I8]: Int8Array,
  [ScalarType.I16]: Int16Array,
  [ScalarType.I32]: Int32Array,
  [ScalarType.I64]: BigInt64Array,
  [ScalarType.UI8]: Uint8Array,
  [ScalarType.BOOL]: Uint8Array,
  [ScalarType.INDEX]: Int32Array,
});

export function typedArrayCtor(dtype) {
  return _TYPED_ARRAY_CTORS[dtype] || Float32Array;
}

const _FLOAT_PRECEDENCE = [ScalarType.F16, ScalarType.BF16, ScalarType.F32, ScalarType.F64];
const _INT_PRECEDENCE = [ScalarType.UI8, ScalarType.I8, ScalarType.I16, ScalarType.I32, ScalarType.I64];

const _PRECEDENCE_MAP = new Map();
for (let i = 0; i < _FLOAT_PRECEDENCE.length; i++) _PRECEDENCE_MAP.set(_FLOAT_PRECEDENCE[i], 100 + i);
for (let i = 0; i < _INT_PRECEDENCE.length; i++) _PRECEDENCE_MAP.set(_INT_PRECEDENCE[i], i);
_PRECEDENCE_MAP.set(ScalarType.BOOL, -1);
_PRECEDENCE_MAP.set(ScalarType.INDEX, 50);

export function resultDtype(a, b) {
  if (a === b) return a;
  const pa = _PRECEDENCE_MAP.get(a) ?? 0;
  const pb = _PRECEDENCE_MAP.get(b) ?? 0;
  return pa >= pb ? a : b;
}

const _CAST_TABLE = new Map();

function _addCastPair(from, to) {
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

export function canCast(from, to) {
  if (from === to) return true;
  const set = _CAST_TABLE.get(from);
  return set ? set.has(to) : false;
}

export function dtypeSize(dtype) {
  return scalarBytes(dtype);
}
