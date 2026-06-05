import { encodeWat } from '../../backend/wasm/wat_encoder.js';

const TYPED_ARRAY_CTORS = {
  'f16':  Float32Array,
  'f32':  Float32Array,
  'f64':  Float64Array,
  'i8':   Int8Array,
  'i16':  Int16Array,
  'i32':  Int32Array,
  'bool': Uint8Array,
  'index': Int32Array,
};

function typedArrayCtor(dtype) {
  return TYPED_ARRAY_CTORS[dtype] || Float32Array;
}

export class RuntimeTensor {
  constructor(data, shape, dtype, strides = null) {
    this.data = data;
    this.shape = shape;
    this.dtype = dtype;
    this.strides = strides || RuntimeTensor.defaultStrides(shape);
  }

  static defaultStrides(shape) {
    const strides = new Array(shape.length);
    let s = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = s;
      s *= shape[i];
    }
    return strides;
  }

  get numel() {
    let n = 1;
    for (let i = 0; i < this.shape.length; i++) n *= this.shape[i];
    return n;
  }

  get rank() {
    return this.shape.length;
  }

  static zeros(shape, dtype = 'f32') {
    let n = 1;
    for (let i = 0; i < shape.length; i++) n *= shape[i];
    n = Math.max(n, 1);
    return new RuntimeTensor(new (typedArrayCtor(dtype))(n), shape, dtype);
  }

  static fromArray(data, shape, dtype = 'f32') {
    const Ctor = typedArrayCtor(dtype);
    return new RuntimeTensor(new Ctor(data), shape, dtype);
  }

  get(indices) {
    let offset = 0;
    for (let i = 0; i < indices.length; i++) offset += indices[i] * this.strides[i];
    return this.data[offset];
  }

  set(indices, value) {
    let offset = 0;
    for (let i = 0; i < indices.length; i++) offset += indices[i] * this.strides[i];
    this.data[offset] = value;
  }
}

export class KernelRegistry {
  constructor() {
    this._kernels = new Map();
  }

  register(name, kernel) { this._kernels.set(name, kernel); }
  get(name) { return this._kernels.get(name) || null; }
  has(name) { return this._kernels.has(name); }
  names() { return [...this._kernels.keys()]; }
}

const SCOPE_PRIORITY = { 'register': 0, 'local': 1, 'shared': 2, 'global': 3 };

export class RuntimeMemoryManager {
  constructor() {
    this._pools = new Map();
    this._allocated = 0;
    this._peak = 0;
  }

  allocate(sizeBytes, dtype = 'f32', scope = 'global') {
    const Ctor = typedArrayCtor(dtype);
    const bytesPerElement = Ctor.BYTES_PER_ELEMENT;
    const count = Math.ceil(sizeBytes / bytesPerElement);
    const poolKey = dtype + ':' + scope;
    let pool = this._pools.get(poolKey);
    if (pool && pool.length > 0) {
      const buf = pool.pop();
      if (buf.length >= count) {
        this._allocated += sizeBytes;
        if (this._allocated > this._peak) this._peak = this._allocated;
        return buf;
      }
    }
    const data = new Ctor(count);
    this._allocated += sizeBytes;
    if (this._allocated > this._peak) this._peak = this._allocated;
    return data;
  }

  release(data, sizeBytes, dtype = 'f32', scope = 'global') {
    this._allocated -= sizeBytes;
    const poolKey = dtype + ':' + scope;
    let pool = this._pools.get(poolKey);
    if (!pool) { pool = []; this._pools.set(poolKey, pool); }
    if (pool.length < 32) pool.push(data);
  }

  get peakUsage() { return this._peak; }
  get currentUsage() { return this._allocated; }
}

const MATH_IMPORTS = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos,
  tan: Math.tan, tanh: Math.tanh, pow: Math.pow, fmod: (a, b) => a % b,
  rsqrt: x => 1 / Math.sqrt(x), sign: Math.sign, round: Math.round,
};

function instantiateWasm(kernel) {
  const binary = encodeWat(kernel.source);
  const mod = new WebAssembly.Module(binary);
  const mathImports = {};
  if (kernel.metadata.imports) {
    for (const [name] of kernel.metadata.imports) {
      mathImports[name] = MATH_IMPORTS[name] || Math[name] || (x => x);
    }
  }
  const instance = new WebAssembly.Instance(mod, { math: mathImports });
  return {
    exports: instance.exports,
    memory: instance.exports.memory,
    bufferOffsets: kernel.metadata.bufferOffsets,
    funcName: kernel.name,
  };
}

function runWasmKernel(wasmInstance, name, tensorArgs, shapeValues) {
  const { exports, memory, bufferOffsets } = wasmInstance;
  const fn = exports[name];
  const offsets = [...bufferOffsets.values()];
  const nBufs = Math.min(offsets.length, tensorArgs.length);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (data instanceof Float32Array) {
      new Float32Array(memory.buffer, offsets[i], data.length).set(data);
    }
  }

  const callArgs = offsets.slice(0, nBufs);
  if (shapeValues) {
    for (const v of shapeValues) callArgs.push(v);
  }
  fn(...callArgs);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (data instanceof Float32Array) {
      data.set(new Float32Array(memory.buffer, offsets[i], data.length));
    }
  }
}

export class WasmTensorPool {
  constructor(wasmInstance) {
    this._inst = wasmInstance;
    this._views = new Map();
  }

  bind(slotIndex, tensor) {
    const offsets = [...this._inst.bufferOffsets.values()];
    const offset = offsets[slotIndex];
    const mem = this._inst.memory;
    new Float32Array(mem.buffer, offset, tensor.data.length).set(tensor.data);
    const view = new Float32Array(mem.buffer, offset, tensor.data.length);
    this._views.set(slotIndex, { tensor, view, offset });
    return view;
  }

  sync(slotIndex) {
    const entry = this._views.get(slotIndex);
    if (entry) entry.tensor.data.set(entry.view);
  }

  syncAll() {
    for (const [, entry] of this._views) entry.tensor.data.set(entry.view);
  }

  runDirect(name, shapeValues) {
    const { exports, bufferOffsets } = this._inst;
    const offsets = [...bufferOffsets.values()];
    const callArgs = [...offsets];
    if (shapeValues) {
      for (const v of shapeValues) callArgs.push(v);
    }
    exports[name](...callArgs);
  }
}

export class RuntimeModule {
  constructor(name) {
    this.name = name;
    this.kernels = new KernelRegistry();
    this.memory = new RuntimeMemoryManager();
    this._compiledFuncs = new Map();
    this._wasmInstances = new Map();
  }

  addCompiledKernel(compiledKernel) {
    this.kernels.register(compiledKernel.name, compiledKernel);
    if (compiledKernel.metadata.kind === 'js') {
      const fn = new Function('return ' + compiledKernel.source)();
      this._compiledFuncs.set(compiledKernel.name, fn);
    } else if (compiledKernel.metadata.kind === 'wasm') {
      const inst = instantiateWasm(compiledKernel);
      this._wasmInstances.set(compiledKernel.name, inst);
    }
  }

  setShapeParamMap(name, shapeParamMap, bufferMap) {
    if (!this._shapeParamMaps) this._shapeParamMaps = new Map();
    this._shapeParamMaps.set(name, shapeParamMap);
    if (bufferMap) {
      if (!this._bufferMaps) this._bufferMaps = new Map();
      this._bufferMaps.set(name, bufferMap);
    }
  }

  run(name, ...args) {
    const tensorArgs = [];
    const tensorShapes = new Map();
    for (let i = 0; i < args.length; i++) {
      if (args[i] instanceof RuntimeTensor) {
        tensorArgs.push(args[i].data);
        tensorShapes.set(i, args[i].shape);
      } else {
        tensorArgs.push(args[i]);
      }
    }

    const shapeParamMap = this._shapeParamMaps && this._shapeParamMaps.get(name);
    let shapeValues = null;
    if (shapeParamMap && shapeParamMap.size > 0) {
      shapeValues = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, args);
    }

    const wasmInst = this._wasmInstances.get(name);
    if (wasmInst) {
      runWasmKernel(wasmInst, name, tensorArgs, shapeValues);
      return;
    }

    const fn = this._compiledFuncs.get(name);
    if (fn) {
      const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
      return fn(...callArgs);
    }

    throw new Error('Kernel \'' + name + '\' not found or not executable');
  }

  static _extractShapeParams(shapeParamMap, tensorShapes, args) {
    const seen = new Map();
    const result = [];
    for (const [key, varNode] of shapeParamMap) {
      if (seen.has(varNode.name)) continue;
      seen.set(varNode.name, true);
      const sepIdx = key.lastIndexOf(':');
      const dimIdx = parseInt(key.substring(sepIdx + 1), 10);
      let resolved = null;
      for (const [, shape] of tensorShapes) {
        if (dimIdx < shape.length && shape[dimIdx] > 0) {
          resolved = shape[dimIdx];
          break;
        }
      }
      result.push(resolved !== null ? resolved : 1);
    }
    return result;
  }

  getWasmPool(name) {
    const inst = this._wasmInstances.get(name);
    if (!inst) return null;
    return new WasmTensorPool(inst);
  }

  getKernelSource(name) {
    const kernel = this.kernels.get(name);
    return kernel ? kernel.source : null;
  }

  listKernels() {
    return this.kernels.names();
  }

  serialize() {
    const entries = [];
    for (const name of this.kernels.names()) {
      const k = this.kernels.get(name);
      entries.push({ name: k.name, source: k.source, target: k.target.name, metadata: k.metadata });
    }
    return { name: this.name, kernels: entries };
  }
}
