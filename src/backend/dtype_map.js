const DTYPE_TABLE = {
  'f16':  { js: 'Float32Array', c: '__half',     cPtr: '__half*',      bytes: 2, suffix: 'h', mathSuffix: 'h',  isFloat: true,  isInt: false },
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

const WASM_TYPE_TABLE = {
  'f16':  { wasm: 'f32', load: 'f32.load', store: 'f32.store', bytes: 4 },
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

export function wasmType(dtype) { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).wasm; }
export function wasmLoad(dtype) { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).load; }
export function wasmStore(dtype) { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).store; }
export function wasmBytes(dtype) { return (WASM_TYPE_TABLE[dtype] || DEFAULT_WASM_ENTRY).bytes; }

const DEFAULT_ENTRY = DTYPE_TABLE['f32'];

export function dtypeInfo(dtype) {
  return DTYPE_TABLE[dtype] || DEFAULT_ENTRY;
}

export function jsTypedArray(dtype) {
  return dtypeInfo(dtype).js;
}

export function cType(dtype) {
  return dtypeInfo(dtype).c;
}

export function cPtrType(dtype) {
  return dtypeInfo(dtype).cPtr;
}

export function cLiteralSuffix(dtype) {
  return dtypeInfo(dtype).suffix;
}

export function cMathFuncSuffix(dtype) {
  return dtypeInfo(dtype).mathSuffix;
}

export function dtypeBytes(dtype) {
  return dtypeInfo(dtype).bytes;
}

export function isDtypeFloat(dtype) {
  return dtypeInfo(dtype).isFloat;
}

export function isDtypeInt(dtype) {
  return dtypeInfo(dtype).isInt;
}

const C_MATH_BASES = {
  'exp': 'exp', 'log': 'log', 'sqrt': 'sqrt', 'tanh': 'tanh',
  'abs': 'fabs', 'sin': 'sin', 'cos': 'cos', 'ceil': 'ceil',
  'floor': 'floor', 'max': 'fmax', 'min': 'fmin', 'pow': 'pow',
  'round': 'round', 'fmod': 'fmod'
};

export function cMathFunc(name, dtype) {
  const base = C_MATH_BASES[name];
  if (!base) return name;
  return base + cMathFuncSuffix(dtype);
}

const JS_MATH_FUNCS = new Set([
  'exp', 'log', 'sqrt', 'tanh', 'abs', 'ceil', 'floor',
  'sin', 'cos', 'max', 'min', 'pow', 'round', 'sign'
]);

export function isJSMathFunc(name) {
  return JS_MATH_FUNCS.has(name);
}

const WGSL_TYPE_TABLE = {
  'f16':  { wgsl: 'f16',  bytes: 2 },
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

export function wgslType(dtype) { return (WGSL_TYPE_TABLE[dtype] || DEFAULT_WGSL_ENTRY).wgsl; }
export function wgslBytes(dtype) { return (WGSL_TYPE_TABLE[dtype] || DEFAULT_WGSL_ENTRY).bytes; }

const WGSL_MATH_FUNCS = {
  'exp': 'exp', 'log': 'log', 'sqrt': 'sqrt', 'tanh': 'tanh',
  'abs': 'abs', 'sin': 'sin', 'cos': 'cos', 'ceil': 'ceil',
  'floor': 'floor', 'max': 'max', 'min': 'min', 'pow': 'pow',
  'round': 'round', 'sign': 'sign', 'rsqrt': 'inverseSqrt',
  'fabs': 'abs',
};

export function wgslMathFunc(name) {
  return WGSL_MATH_FUNCS[name] || name;
}

const LIBRARY_FUNC_TABLE = {
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

export function libraryFunc(opName, libraryName, dtype) {
  const opEntry = LIBRARY_FUNC_TABLE[opName];
  if (!opEntry) return null;
  const libEntry = opEntry[libraryName];
  if (!libEntry) return null;
  return libEntry[dtype] || null;
}
