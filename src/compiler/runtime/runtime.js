import { encodeWat } from '../../backend/wasm/wat_encoder.js';
let _webgpuMod = null;
async function getWebGPURuntime() {
  if (!_webgpuMod) _webgpuMod = await import('./webgpu_runtime.js');
  return _webgpuMod;
}

let _wasmPoolMod = null;
async function getWasmPool() {
  if (!_wasmPoolMod) _wasmPoolMod = await import('#io/wasm_pool');
  return _wasmPoolMod;
}

const TYPED_ARRAY_CTORS = {
  'f16':  Uint16Array,
  'bf16': Uint16Array,
  'f32':  Float32Array,
  'f64':  Float64Array,
  'i8':   Int8Array,
  'i16':  Int16Array,
  'i32':  Int32Array,
  'i64':  BigInt64Array,
  'ui8':  Uint8Array,
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
      let fitIdx = -1;
      for (let i = 0; i < pool.length; i++) {
        if (pool[i].length >= count && (fitIdx === -1 || pool[i].length < pool[fitIdx].length)) {
          fitIdx = i;
        }
      }
      if (fitIdx !== -1) {
        const buf = pool[fitIdx];
        pool[fitIdx] = pool[pool.length - 1];
        pool.pop();
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
  let binary;
  try { binary = encodeWat(kernel.source); } catch(e) { throw new Error('encodeWat: ' + e.message + '\n' + kernel.source); }
  let mod;
  try { mod = new WebAssembly.Module(binary); } catch(e) { throw new Error('WASM: ' + e.message + '\n' + kernel.source); }
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
    binary,
  };
}

function runWasmKernel(wasmInstance, name, tensorArgs, shapeValues, parStart, parEnd) {
  const { exports, memory, bufferOffsets } = wasmInstance;
  const fn = exports[name];
  const offsets = [...bufferOffsets.values()];
  const nBufs = Math.min(offsets.length, tensorArgs.length);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      new data.constructor(memory.buffer, offsets[i], data.length).set(data);
    }
  }

  const callArgs = offsets.slice(0, nBufs);
  if (shapeValues) {
    for (const v of shapeValues) callArgs.push(v);
  }
  if (parStart !== undefined && parEnd !== undefined) {
    callArgs.push(parStart, parEnd);
  }
  fn(...callArgs);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      data.set(new data.constructor(memory.buffer, offsets[i], data.length));
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
      if (compiledKernel.metadata.parallel) {
        if (!this._wasmParallel) this._wasmParallel = new Map();
        this._wasmParallel.set(compiledKernel.name, compiledKernel.metadata.parallel);
      }
    } else if (compiledKernel.metadata.kind === 'webgpu') {
      if (!this._webgpuKernels) this._webgpuKernels = new Map();
      if (!this._webgpuInstances) this._webgpuInstances = new Map();
      this._webgpuKernels.set(compiledKernel.name, compiledKernel);
      this._webgpuInstances.set(compiledKernel.name, getWebGPURuntime().then(m => m.instantiateWebGPU(compiledKernel)));
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
      const bufferMap = this._bufferMaps && this._bufferMaps.get(name);
      shapeValues = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, args, bufferMap);
    }

    if (this._webgpuKernels && this._webgpuKernels.has(name)) {
      throw new Error('WebGPU kernel requires async execution — use runAsync()');
    }

    const wasmInst = this._wasmInstances.get(name);
    if (wasmInst) {
      const parallel = this._wasmParallel && this._wasmParallel.get(name);
      if (parallel) {
        runWasmKernel(wasmInst, name, tensorArgs, shapeValues, 0, parallel.extent);
      } else {
        runWasmKernel(wasmInst, name, tensorArgs, shapeValues);
      }
      return;
    }

    const fn = this._compiledFuncs.get(name);
    if (fn) {
      const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
      return fn(...callArgs);
    }

    throw new Error('Kernel \'' + name + '\' not found or not executable');
  }

  async runAsync(name, ...args) {
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
      const bufferMap = this._bufferMaps && this._bufferMaps.get(name);
      shapeValues = RuntimeModule._extractShapeParams(shapeParamMap, tensorShapes, args, bufferMap);
    }

    if (this._wasmParallel && this._wasmParallel.has(name) && this._wasmParallel.get(name).poolSafe) {
      const wasmInst = this._wasmInstances.get(name);
      const parallel = this._wasmParallel.get(name);
      const kernel = this.kernels.get(name);
      const mathNames = kernel.metadata.imports ? [...kernel.metadata.imports.keys()] : [];
      const { runWasmParallel: runPar } = await getWasmPool();
      await runPar(wasmInst, name, tensorArgs, shapeValues, parallel, mathNames);
      return;
    }

    if (this._webgpuInstances && this._webgpuInstances.has(name)) {
      const instance = await this._webgpuInstances.get(name);
      const { runWebGPUKernel } = await getWebGPURuntime();
      await runWebGPUKernel(instance, tensorArgs, shapeValues);
      return;
    }

    this.run(name, ...args);
  }

  isAsync(name) {
    return !!(this._webgpuKernels && this._webgpuKernels.has(name))
      || !!(this._wasmParallel && this._wasmParallel.has(name) && this._wasmParallel.get(name).poolSafe);
  }

  static _extractShapeParams(shapeParamMap, tensorShapes, args, bufferMap) {
    const bufferIndex = new Map();
    if (bufferMap) {
      let i = 0;
      for (const [k, buf] of bufferMap) {
        const name = typeof k === 'string' ? k : (buf && buf.name);
        if (name !== undefined) bufferIndex.set(name, i);
        i++;
      }
    }
    const seen = new Map();
    const result = [];
    for (const [key, varNode] of shapeParamMap) {
      if (seen.has(varNode.name)) continue;
      seen.set(varNode.name, true);
      const sepIdx = key.lastIndexOf(':');
      const bufferName = key.substring(0, sepIdx);
      const dimIdx = parseInt(key.substring(sepIdx + 1), 10);
      let resolved = null;
      if (bufferIndex.has(bufferName)) {
        const shape = tensorShapes.get(bufferIndex.get(bufferName));
        if (shape && dimIdx < shape.length && shape[dimIdx] > 0) {
          resolved = shape[dimIdx];
        }
      }
      if (resolved === null) {
        for (const [, shape] of tensorShapes) {
          if (dimIdx < shape.length && shape[dimIdx] > 0) {
            resolved = shape[dimIdx];
            break;
          }
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

  getKernelSnippet(name) {
    const kernel = this.kernels.get(name);
    return kernel ? kernel.snippet() : null;
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
