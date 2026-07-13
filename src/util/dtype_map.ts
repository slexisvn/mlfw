type DTypeInfo = {
  js: string;
  c: string;
  cPtr: string;
  bytes: number;
  suffix: string;
  mathSuffix: string;
  isFloat: boolean;
  isInt: boolean;
};

type WasmTypeInfo = {
  wasm: string;
  load: string;
  store: string;
  bytes: number;
};

type SimdInfo = {
  laneType: string;
  lanes: number;
  laneBytes: number;
  vecLoad: string;
  vecStore: string;
  splat: string;
  extractLane: string;
  replaceLane: string;
  add: string;
  sub: string;
  mul: string;
  div: string | null;
  neg: string | null;
  abs: string | null;
  sqrt: string | null;
  min: string | null;
  max: string | null;
  ceil: string | null;
  floor: string | null;
  eq: string;
  ne: string;
  lt: string;
  le: string;
  gt: string;
  ge: string;
  bitselect: string;
};

type WgslTypeInfo = {
  wgsl: string;
  bytes: number;
};

const DTYPE_TABLE: Record<string, DTypeInfo> = {
  'f16':  { js: 'Uint16Array',  c: '__half',     cPtr: '__half*',      bytes: 2, suffix: 'h', mathSuffix: 'h',  isFloat: true,  isInt: false },
  'bf16': { js: 'Uint16Array',  c: '__nv_bfloat16', cPtr: '__nv_bfloat16*', bytes: 2, suffix: '', mathSuffix: '', isFloat: true, isInt: false },
  'f32':  { js: 'Float32Array', c: 'float',      cPtr: 'float*',      bytes: 4, suffix: 'f', mathSuffix: 'f',  isFloat: true,  isInt: false },
  'f64':  { js: 'Float64Array', c: 'double',     cPtr: 'double*',     bytes: 8, suffix: '',  mathSuffix: '',   isFloat: true,  isInt: false },
  'i8':   { js: 'Int8Array',    c: 'int8_t',     cPtr: 'int8_t*',     bytes: 1, suffix: '',  mathSuffix: '',   isFloat: false, isInt: true  },
  'i16':  { js: 'Int16Array',   c: 'int16_t',    cPtr: 'int16_t*',    bytes: 2, suffix: '',  mathSuffix: '',   isFloat: false, isInt: true  },
  'i32':  { js: 'Int32Array',   c: 'int',        cPtr: 'int*',        bytes: 4, suffix: '',  mathSuffix: '',   isFloat: false, isInt: true  },
  'i64':  { js: 'BigInt64Array', c: 'int64_t',   cPtr: 'int64_t*',    bytes: 8, suffix: 'LL', mathSuffix: '',  isFloat: false, isInt: true  },
  'ui8':  { js: 'Uint8Array',   c: 'uint8_t',    cPtr: 'uint8_t*',    bytes: 1, suffix: '',  mathSuffix: '',   isFloat: false, isInt: true  },
  'bool': { js: 'Uint8Array',   c: 'bool',       cPtr: 'bool*',       bytes: 1, suffix: '',  mathSuffix: '',   isFloat: false, isInt: false },
  'index':{ js: 'Int32Array',   c: 'int',        cPtr: 'int*',        bytes: 4, suffix: '',  mathSuffix: '',   isFloat: false, isInt: true  },
};

const WASM_TYPE_TABLE: Record<string, WasmTypeInfo> = {
  'f16':  { wasm: 'f32', load: 'i32.load16_u', store: 'i32.store16', bytes: 2 },
  'bf16': { wasm: 'f32', load: 'i32.load16_u', store: 'i32.store16', bytes: 2 },
  'f32':  { wasm: 'f32', load: 'f32.load', store: 'f32.store', bytes: 4 },
  'f64':  { wasm: 'f64', load: 'f64.load', store: 'f64.store', bytes: 8 },
  'i8':   { wasm: 'i32', load: 'i32.load8_s', store: 'i32.store8', bytes: 1 },
  'i16':  { wasm: 'i32', load: 'i32.load16_s', store: 'i32.store16', bytes: 2 },
  'i32':  { wasm: 'i32', load: 'i32.load', store: 'i32.store', bytes: 4 },
  'i64':  { wasm: 'i64', load: 'i64.load', store: 'i64.store', bytes: 8 },
  'ui8':  { wasm: 'i32', load: 'i32.load8_u', store: 'i32.store8', bytes: 1 },
  'bool': { wasm: 'i32', load: 'i32.load8_u', store: 'i32.store8', bytes: 1 },
  'index':{ wasm: 'i32', load: 'i32.load', store: 'i32.store', bytes: 4 },
};

const DEFAULT_WASM_ENTRY = WASM_TYPE_TABLE['f32'];

const WASM_SIMD_TABLE: Record<string, SimdInfo> = {
  'f32': {
    laneType: 'f32x4', lanes: 4, laneBytes: 4,
    vecLoad: 'v128.load', vecStore: 'v128.store',
    splat: 'f32x4.splat', extractLane: 'f32x4.extract_lane', replaceLane: 'f32x4.replace_lane',
    add: 'f32x4.add', sub: 'f32x4.sub', mul: 'f32x4.mul', div: 'f32x4.div',
    neg: 'f32x4.neg', abs: 'f32x4.abs', sqrt: 'f32x4.sqrt',
    min: 'f32x4.min', max: 'f32x4.max', ceil: 'f32x4.ceil', floor: 'f32x4.floor',
    eq: 'f32x4.eq', ne: 'f32x4.ne', lt: 'f32x4.lt', le: 'f32x4.le', gt: 'f32x4.gt', ge: 'f32x4.ge',
    bitselect: 'v128.bitselect',
  },
  'i32': {
    laneType: 'i32x4', lanes: 4, laneBytes: 4,
    vecLoad: 'v128.load', vecStore: 'v128.store',
    splat: 'i32x4.splat', extractLane: 'i32x4.extract_lane', replaceLane: 'i32x4.replace_lane',
    add: 'i32x4.add', sub: 'i32x4.sub', mul: 'i32x4.mul', div: null,
    neg: null, abs: 'i32x4.abs', sqrt: null,
    min: 'i32x4.min_s', max: 'i32x4.max_s', ceil: null, floor: null,
    eq: 'i32x4.eq', ne: 'i32x4.ne', lt: 'i32x4.lt_s', le: 'i32x4.le_s', gt: 'i32x4.gt_s', ge: 'i32x4.ge_s',
    bitselect: 'v128.bitselect',
  },
};

export function wasmType(dtype: string): string { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).wasm; }
export function wasmLoad(dtype: string): string { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).load; }
export function wasmStore(dtype: string): string { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).store; }
export function wasmBytes(dtype: string): number { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).bytes; }

export function wasmSimdEntry(dtype: string): SimdInfo | null { return WASM_SIMD_TABLE[dtype] || null; }
export function wasmVecOp(dtype: string, op: keyof SimdInfo): string | number | null { const e = WASM_SIMD_TABLE[dtype]; return e ? (e[op] || null) : null; }
export function wasmVecLoad(dtype: string): string | null { const e = WASM_SIMD_TABLE[dtype]; return e ? e.vecLoad : null; }
export function wasmVecStore(dtype: string): string | null { const e = WASM_SIMD_TABLE[dtype]; return e ? e.vecStore : null; }
export function wasmVecSplat(dtype: string): string | null { const e = WASM_SIMD_TABLE[dtype]; return e ? e.splat : null; }
export function wasmVecExtractLane(dtype: string): string | null { const e = WASM_SIMD_TABLE[dtype]; return e ? e.extractLane : null; }
export function wasmVecReplaceLane(dtype: string): string | null { const e = WASM_SIMD_TABLE[dtype]; return e ? e.replaceLane : null; }
export function wasmVecLanes(dtype: string): number { const e = WASM_SIMD_TABLE[dtype]; return e ? e.lanes : 0; }

const DEFAULT_ENTRY = DTYPE_TABLE['f32'];

export function dtypeInfo(dtype: string): DTypeInfo {
  return DTYPE_TABLE[dtype] || DEFAULT_ENTRY;
}

export function jsTypedArray(dtype: string): string {
  return dtypeInfo(dtype).js;
}

export function cType(dtype: string): string {
  return dtypeInfo(dtype).c;
}

export function cPtrType(dtype: string): string {
  return dtypeInfo(dtype).cPtr;
}

export function cLiteralSuffix(dtype: string): string {
  return dtypeInfo(dtype).suffix;
}

export function cMathFuncSuffix(dtype: string): string {
  return dtypeInfo(dtype).mathSuffix;
}

export function dtypeBytes(dtype: string): number {
  return dtypeInfo(dtype).bytes;
}

export function isDtypeFloat(dtype: string): boolean {
  return dtypeInfo(dtype).isFloat;
}

export function isDtypeInt(dtype: string): boolean {
  return dtypeInfo(dtype).isInt;
}

const INT_RANGE: Record<string, readonly [number, number]> = {
  'i8': [-128, 127], 'i16': [-32768, 32767], 'i32': [-2147483648, 2147483647],
  'i64': [Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER],
  'ui8': [0, 255], 'index': [-2147483648, 2147483647],
};

export function reduceInitValue(reduceType: string, dtype: string): number {
  if (reduceType === 'sum' || reduceType === 'mean') return 0;
  if (reduceType === 'prod') return 1;
  if (reduceType === 'max') return isDtypeInt(dtype) ? (INT_RANGE[dtype] || INT_RANGE['i32'])[0] : -Infinity;
  if (reduceType === 'min') return isDtypeInt(dtype) ? (INT_RANGE[dtype] || INT_RANGE['i32'])[1] : Infinity;
  return 0;
}

const C_MATH_BASES: Record<string, string> = {
  'exp': 'exp', 'log': 'log', 'sqrt': 'sqrt', 'tanh': 'tanh',
  'abs': 'fabs', 'sin': 'sin', 'cos': 'cos', 'ceil': 'ceil',
  'floor': 'floor', 'max': 'fmax', 'min': 'fmin', 'pow': 'pow',
  'round': 'round', 'fmod': 'fmod', 'rsqrt': 'rsqrt',
  'erf': 'erf', 'erfc': 'erfc', 'lgamma': 'lgamma', 'gamma': 'tgamma', 'log2': 'log2', 'log10': 'log10', 'exp2': 'exp2'
};

export function cMathFunc(name: string, dtype: string): string {
  const base = C_MATH_BASES[name];
  if (!base) return name;
  return base + cMathFuncSuffix(dtype);
}

const COMPARE_C_OPS: Record<string, string> = { eq: '==', ne: '!=', lt: '<', le: '<=', gt: '>', ge: '>=' };
const COMPARE_JS_OPS: Record<string, string> = { eq: '===', ne: '!==', lt: '<', le: '<=', gt: '>', ge: '>=' };

export function cCompareOp(direction: string): string {
  const op = COMPARE_C_OPS[direction];
  if (!op) throw new Error(`unsupported compare direction '${direction}'`);
  return op;
}

export function jsCompareOp(direction: string): string {
  const op = COMPARE_JS_OPS[direction];
  if (!op) throw new Error(`unsupported compare direction '${direction}'`);
  return op;
}

const JS_MATH_FUNCS = new Set([
  'exp', 'log', 'sqrt', 'tanh', 'abs', 'ceil', 'floor',
  'sin', 'cos', 'max', 'min', 'pow', 'round', 'sign',
  'log2', 'log10'
]);

export function isJSMathFunc(name: string): boolean {
  return JS_MATH_FUNCS.has(name);
}

const WGSL_TYPE_TABLE: Record<string, WgslTypeInfo> = {
  'f16':  { wgsl: 'f16',  bytes: 2 },
  'bf16': { wgsl: 'f32',  bytes: 4 },
  'f32':  { wgsl: 'f32',  bytes: 4 },
  'f64':  { wgsl: 'f32',  bytes: 4 },
  'i8':   { wgsl: 'i32',  bytes: 4 },
  'i16':  { wgsl: 'i32',  bytes: 4 },
  'i32':  { wgsl: 'i32',  bytes: 4 },
  'i64':  { wgsl: 'i32',  bytes: 4 },
  'ui8':  { wgsl: 'u32',  bytes: 4 },
  'bool': { wgsl: 'u32',  bytes: 4 },
  'index':{ wgsl: 'u32',  bytes: 4 },
};

const DEFAULT_WGSL_ENTRY = WGSL_TYPE_TABLE['f32'];

export function wgslType(dtype: string): string { return (WGSL_TYPE_TABLE[dtype] || DEFAULT_WGSL_ENTRY).wgsl; }
export function wgslBytes(dtype: string): number { return (WGSL_TYPE_TABLE[dtype] || DEFAULT_WGSL_ENTRY).bytes; }

const WGSL_MATH_FUNCS: Record<string, string> = {
  'exp': 'exp', 'log': 'log', 'sqrt': 'sqrt', 'tanh': 'tanh',
  'abs': 'abs', 'sin': 'sin', 'cos': 'cos', 'ceil': 'ceil',
  'floor': 'floor', 'max': 'max', 'min': 'min', 'pow': 'pow',
  'round': 'round', 'sign': 'sign', 'rsqrt': 'inverseSqrt',
  'fabs': 'abs', 'log2': 'log2', 'exp2': 'exp2',
};

export function wgslMathFunc(name: string): string {
  return WGSL_MATH_FUNCS[name] || name;
}

export function hasWgslMathFunc(name: string): boolean {
  return name in WGSL_MATH_FUNCS;
}

const LIBRARY_FUNC_TABLE: Record<string, Record<string, Record<string, string>>> = {
  'dot': {
    'blas':   { 'f32': 'sgemm',  'f64': 'dgemm'  },
    'cublas': { 'f32': 'cublasSgemm', 'f64': 'cublasDgemm', 'f16': 'cublasHgemm' },
  },
  'conv': {
    'cudnn': { 'f32': 'cudnnConvolutionForward', 'f16': 'cudnnConvolutionForward' },
  },
  'quantized_dot': {
    'blas':   { 'i8': 'cblas_gemm_s8u8s32' },
    'cublas': { 'i8': 'cublasGemmEx' },
  },
  'quantized_conv': {
    'cudnn': { 'i8': 'cudnnConvolutionForward' },
  }
};

export function libraryFunc(opName: string, libraryName: string, dtype: string): string | null {
  const opEntry = LIBRARY_FUNC_TABLE[opName];
  if (!opEntry) return null;
  const libEntry = opEntry[libraryName];
  if (!libEntry) return null;
  return libEntry[dtype] || null;
}
