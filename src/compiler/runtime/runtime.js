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

  register(name, kernel) {
    this._kernels.set(name, kernel);
  }

  get(name) {
    return this._kernels.get(name) || null;
  }

  has(name) {
    return this._kernels.has(name);
  }

  names() {
    return [...this._kernels.keys()];
  }
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
    const poolKey = `${dtype}:${scope}`;
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
    const poolKey = `${dtype}:${scope}`;
    let pool = this._pools.get(poolKey);
    if (!pool) {
      pool = [];
      this._pools.set(poolKey, pool);
    }
    if (pool.length < 32) pool.push(data);
  }

  get peakUsage() { return this._peak; }
  get currentUsage() { return this._allocated; }
}

export class RuntimeModule {
  constructor(name) {
    this.name = name;
    this.kernels = new KernelRegistry();
    this.memory = new RuntimeMemoryManager();
    this._compiledFuncs = new Map();
    this._compileErrors = null;
  }

  addCompiledKernel(compiledKernel) {
    this.kernels.register(compiledKernel.name, compiledKernel);
    if (compiledKernel.metadata.kind === 'js') {
      try {
        const fn = new Function('return ' + compiledKernel.source)();
        this._compiledFuncs.set(compiledKernel.name, fn);
      } catch (e) {
        if (!this._compileErrors) this._compileErrors = new Map();
        this._compileErrors.set(compiledKernel.name, e.message);
        this._compiledFuncs.set(compiledKernel.name, null);
      }
    }
  }

  run(name, ...args) {
    const fn = this._compiledFuncs.get(name);
    if (!fn) {
      const err = this._compileErrors && this._compileErrors.get(name);
      throw new Error(`Kernel '${name}' not found or not executable${err ? ': ' + err : ''}`);
    }
    const tensorArgs = new Array(args.length);
    for (let i = 0; i < args.length; i++) {
      tensorArgs[i] = args[i] instanceof RuntimeTensor ? args[i].data : args[i];
    }
    return fn(...tensorArgs);
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
