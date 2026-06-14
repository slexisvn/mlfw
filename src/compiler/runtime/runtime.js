import { getBackend } from './backend_registry.js';

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

function dtypeOfTypedArray(a) {
  if (a instanceof Float64Array) return 'f64';
  if (a instanceof Int32Array) return 'i32';
  if (a instanceof Int16Array) return 'i16';
  if (a instanceof Int8Array) return 'i8';
  if (a instanceof Uint16Array) return 'ui16';
  if (a instanceof Uint8Array) return 'ui8';
  if (a instanceof BigInt64Array) return 'i64';
  return 'f32';
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

export class RuntimeModule {
  constructor(name) {
    this.name = name;
    this.kernels = new KernelRegistry();
    this._instances = new Map();
  }

  addCompiledKernel(compiledKernel) {
    this.kernels.register(compiledKernel.name, compiledKernel);
    const backend = getBackend(compiledKernel.metadata.kind);
    if (!backend) throw new Error('No runtime backend registered for kind: ' + compiledKernel.metadata.kind);
    this._instances.set(compiledKernel.name, { backend, instance: backend.instantiate(compiledKernel) });
  }

  setShapeParamMap(name, shapeParamMap, bufferMap) {
    if (!this._shapeParamMaps) this._shapeParamMaps = new Map();
    this._shapeParamMaps.set(name, shapeParamMap);
    if (bufferMap) {
      if (!this._bufferMaps) this._bufferMaps = new Map();
      this._bufferMaps.set(name, bufferMap);
    }
  }

  _prepareArgs(name, args) {
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
    return { tensorArgs, shapeValues };
  }

  run(name, ...args) {
    const entry = this._instances.get(name);
    if (!entry) throw new Error('Kernel \'' + name + '\' not found or not executable');
    if (entry.instance instanceof Promise) {
      throw new Error('Kernel \'' + name + '\' requires async execution — use runAsync()');
    }
    const { tensorArgs, shapeValues } = this._prepareArgs(name, args);
    return entry.backend.runSync(entry.instance, tensorArgs, shapeValues);
  }

  async runAsync(name, ...args) {
    const entry = this._instances.get(name);
    if (!entry) throw new Error('Kernel \'' + name + '\' not found or not executable');
    const { tensorArgs, shapeValues } = this._prepareArgs(name, args);
    const instance = await entry.instance;
    return entry.backend.runAsync(instance, tensorArgs, shapeValues);
  }

  isAsync(name) {
    const entry = this._instances.get(name);
    if (!entry) return false;
    const inst = entry.instance instanceof Promise ? null : entry.instance;
    return entry.backend.isAsync(inst);
  }

  async runPlanAsync(plan, args) {
    const slots = new Array(plan.numSlots).fill(null);
    for (let i = 0; i < args.length; i++) {
      const a = args[i];
      slots[plan.argSlots[i]] = a instanceof RuntimeTensor ? a
        : new RuntimeTensor(a, [a.length], dtypeOfTypedArray(a));
    }
    for (const it of plan.intermediates) {
      let numel = 1;
      for (const d of it.shape) numel *= d;
      slots[it.slot] = new RuntimeTensor(new (typedArrayCtor(it.dtype))(Math.max(numel, 1)), it.shape, it.dtype);
    }

    const planBackend = this._uniformPlanBackend(plan);
    if (planBackend && planBackend.runPlan) {
      const steps = plan.steps.map(step => {
        const stepArgs = [];
        for (const s of step.inputSlots) stepArgs.push(slots[s]);
        for (const s of step.outputSlots) stepArgs.push(slots[s]);
        const { shapeValues } = this._prepareArgs(step.name, stepArgs);
        return { name: step.name, inputSlots: step.inputSlots, outputSlots: step.outputSlots, kernel: this.kernels.get(step.name), shapeValues };
      });
      return planBackend.runPlan(plan, slots, steps);
    }

    for (const step of plan.steps) {
      const stepArgs = [];
      for (const s of step.inputSlots) stepArgs.push(slots[s]);
      for (const s of step.outputSlots) stepArgs.push(slots[s]);
      await this.runAsync(step.name, ...stepArgs);
    }
  }

  _uniformPlanBackend(plan) {
    let backend = null;
    for (const step of plan.steps) {
      const entry = this._instances.get(step.name);
      if (!entry || entry.instance instanceof Promise) return null;
      if (backend === null) backend = entry.backend;
      else if (entry.backend !== backend) return null;
    }
    return backend;
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
