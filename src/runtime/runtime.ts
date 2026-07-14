import { getBackend } from './backend_registry.js';
import { CompiledKernel } from '../backend/pipeline.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';

type RuntimeDType = string;
type RuntimeData = NumericTypedArray;
type RuntimeArrayConstructor = {
  new(length: number): RuntimeData;
  new(data: ArrayLike<number> | ArrayLike<bigint>): RuntimeData;
};
type RuntimeArg = RuntimeTensor | RuntimeData;
type ShapeParamMap = Map<string, { name: string }>;
type BufferMap = Map<string | { name?: string }, { name?: string } | null | undefined>;
type RuntimeKernel = {
  name: string;
  source: string;
  target: { name: string };
  metadata: {
    kind: string;
    imports?: Map<string, unknown>;
    bufferOffsets?: Map<unknown, number>;
    scratch?: unknown[];
    parallel?: { extent: number; poolSafe?: boolean } | null;
    [key: string]: unknown;
  };
  snippet(): string | null;
};
type RuntimeBackend = {
  instantiate(kernel: RuntimeKernel): unknown | Promise<unknown>;
  runSync(instance: unknown, tensorArgs: unknown[], shapeValues: number[] | null): unknown;
  runAsync(instance: unknown, tensorArgs: unknown[], shapeValues: number[] | null): Promise<unknown>;
  runPlan?: (plan: ExecutionPlan, slots: Array<RuntimeTensor | null>, steps: PlanStepRuntime[], opts?: Record<string, unknown>) => Promise<unknown>;
  isAsync(instance: unknown): boolean;
};
type KernelInstanceEntry = { backend: RuntimeBackend; instance: unknown | Promise<unknown> };
type ReturnFixup = { pos: number; kind: 'copy'; srcSlot: number } | { pos: number; kind: 'const'; value: number };
type ExecutionPlanStep = { name: string; inputSlots: number[]; outputSlots: number[] };
type ExecutionPlan = {
  numSlots: number;
  argSlots: number[];
  intermediates: Array<{ slot: number; shape: number[]; dtype: RuntimeDType }>;
  steps: ExecutionPlanStep[];
  returnFixups?: ReturnFixup[];
};
type PlanStepRuntime = ExecutionPlanStep & {
  kernel: RuntimeKernel | null;
  shapeValues: number[] | null;
};
type SerializedRuntimeModule = {
  name: string;
  kernels: Array<{ name: string; source: string; target: string; metadata: RuntimeKernel['metadata'] }>;
};

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

function typedArrayCtor(dtype: RuntimeDType): RuntimeArrayConstructor {
  return (TYPED_ARRAY_CTORS[dtype as keyof typeof TYPED_ARRAY_CTORS] || Float32Array) as RuntimeArrayConstructor;
}

function dtypeOfTypedArray(a: RuntimeData): RuntimeDType {
  if (a instanceof Float64Array) return 'f64';
  if (a instanceof Int32Array) return 'i32';
  if (a instanceof Int16Array) return 'i16';
  if (a instanceof Int8Array) return 'i8';
  if (a instanceof Uint16Array) return 'ui16';
  if (a instanceof Uint8Array) return 'ui8';
  if (a instanceof BigInt64Array) return 'i64';
  return 'f32';
}

function _applyReturnFixups(plan: ExecutionPlan, slots: Array<RuntimeTensor | null>): void {
  if (!plan.returnFixups || plan.returnFixups.length === 0) return;
  for (const fx of plan.returnFixups) {
    const dst = slots[plan.argSlots[fx.pos]];
    if (!dst || !dst.data) continue;
    if (fx.kind === 'copy') {
      const src = slots[fx.srcSlot];
      if (src && src.data) (dst.data as { set(values: unknown): void }).set(src.data.subarray(0, dst.data.length));
    } else if (fx.kind === 'const') {
      (dst.data as { fill(value: number): void }).fill(fx.value);
    }
  }
}

export class RuntimeTensor {
  data: RuntimeData;
  shape: number[];
  dtype: RuntimeDType;
  strides: number[];
  resident?: { key: unknown; version: number };

  constructor(data: RuntimeData, shape: readonly number[], dtype: RuntimeDType, strides: readonly number[] | null = null) {
    this.data = data;
    this.shape = [...shape];
    this.dtype = dtype;
    this.strides = strides ? [...strides] : RuntimeTensor.defaultStrides(shape);
  }

  static defaultStrides(shape: readonly number[]): number[] {
    const strides = new Array<number>(shape.length);
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

  static zeros(shape: readonly number[], dtype: RuntimeDType = 'f32'): RuntimeTensor {
    let n = 1;
    for (let i = 0; i < shape.length; i++) n *= shape[i];
    n = Math.max(n, 1);
    return new RuntimeTensor(new (typedArrayCtor(dtype))(n), shape, dtype);
  }

  static fromArray(data: ArrayLike<number>, shape: readonly number[], dtype: RuntimeDType = 'f32'): RuntimeTensor {
    const Ctor = typedArrayCtor(dtype);
    return new RuntimeTensor(new Ctor(data), shape, dtype);
  }

  get(indices: readonly number[]): number | bigint {
    let offset = 0;
    for (let i = 0; i < indices.length; i++) offset += indices[i] * this.strides[i];
    return this.data[offset];
  }

  set(indices: readonly number[], value: number | bigint): void {
    let offset = 0;
    for (let i = 0; i < indices.length; i++) offset += indices[i] * this.strides[i];
    (this.data as { [index: number]: number | bigint })[offset] = value;
  }
}

export class KernelRegistry {
  private _kernels: Map<string, RuntimeKernel>;

  constructor() {
    this._kernels = new Map();
  }

  register(name: string, kernel: RuntimeKernel): void { this._kernels.set(name, kernel); }
  get(name: string): RuntimeKernel | null { return this._kernels.get(name) || null; }
  has(name: string): boolean { return this._kernels.has(name); }
  names(): string[] { return [...this._kernels.keys()]; }
}

export class RuntimeModule {
  name: string;
  kernels: KernelRegistry;
  private _instances: Map<string, KernelInstanceEntry>;
  private _shapeParamMaps?: Map<string, ShapeParamMap>;
  private _bufferMaps?: Map<string, BufferMap>;

  constructor(name: string) {
    this.name = name;
    this.kernels = new KernelRegistry();
    this._instances = new Map();
  }

  addCompiledKernel(compiledKernel: RuntimeKernel): void {
    this.kernels.register(compiledKernel.name, compiledKernel);
    const backend = getBackend(compiledKernel.metadata.kind) as RuntimeBackend | null;
    if (!backend) throw new Error('No runtime backend registered for kind: ' + compiledKernel.metadata.kind);
    this._instances.set(compiledKernel.name, { backend, instance: backend.instantiate(compiledKernel) });
  }

  setShapeParamMap(name: string, shapeParamMap: ShapeParamMap, bufferMap?: BufferMap): void {
    if (!this._shapeParamMaps) this._shapeParamMaps = new Map();
    this._shapeParamMaps.set(name, shapeParamMap);
    if (bufferMap) {
      if (!this._bufferMaps) this._bufferMaps = new Map();
      this._bufferMaps.set(name, bufferMap);
    }
  }

  _prepareArgs(name: string, args: readonly RuntimeArg[]): { tensorArgs: unknown[]; shapeValues: number[] | null } {
    const tensorArgs: unknown[] = [];
    const tensorShapes = new Map<number, number[]>();
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      if (arg instanceof RuntimeTensor) {
        tensorArgs.push(arg.data);
        tensorShapes.set(i, arg.shape);
      } else {
        tensorArgs.push(arg);
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

  run(name: string, ...args: RuntimeArg[]): unknown {
    const entry = this._instances.get(name);
    if (!entry) throw new Error('Kernel \'' + name + '\' not found or not executable');
    if (entry.instance instanceof Promise) {
      throw new Error('Kernel \'' + name + '\' requires async execution — use runAsync()');
    }
    const { tensorArgs, shapeValues } = this._prepareArgs(name, args);
    return entry.backend.runSync(entry.instance, tensorArgs, shapeValues);
  }

  async runAsync(name: string, ...args: RuntimeArg[]): Promise<unknown> {
    const entry = this._instances.get(name);
    if (!entry) throw new Error('Kernel \'' + name + '\' not found or not executable');
    const { tensorArgs, shapeValues } = this._prepareArgs(name, args);
    const instance = await entry.instance;
    return entry.backend.runAsync(instance, tensorArgs, shapeValues);
  }

  isAsync(name: string): boolean {
    const entry = this._instances.get(name);
    if (!entry) return false;
    const inst = entry.instance instanceof Promise ? null : entry.instance;
    return entry.backend.isAsync(inst);
  }

  async runPlanAsync(plan: ExecutionPlan, args: RuntimeArg[], opts?: Record<string, unknown>): Promise<void> {
    const slots = new Array<RuntimeTensor | null>(plan.numSlots).fill(null);
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

    for (const step of plan.steps) {
      const entry = this._instances.get(step.name);
      if (entry && entry.instance instanceof Promise) entry.instance = await entry.instance;
    }

    const planBackend = this._uniformPlanBackend(plan);
    const anyScratch = plan.steps.some(step => {
      const k = this.kernels.get(step.name);
      return k && k.metadata && k.metadata.scratch && k.metadata.scratch.length > 0;
    });
    if (planBackend && planBackend.runPlan && !anyScratch) {
      const steps = plan.steps.map((step): PlanStepRuntime => {
        const stepArgs: RuntimeArg[] = [];
        for (const s of step.inputSlots) stepArgs.push(slots[s]!);
        for (const s of step.outputSlots) stepArgs.push(slots[s]!);
        const { shapeValues } = this._prepareArgs(step.name, stepArgs);
        return { name: step.name, inputSlots: step.inputSlots, outputSlots: step.outputSlots, kernel: this.kernels.get(step.name), shapeValues };
      });
      await planBackend.runPlan(plan, slots, steps, opts);
      _applyReturnFixups(plan, slots);
      return;
    }

    for (const step of plan.steps) {
      const stepArgs: RuntimeArg[] = [];
      for (const s of step.inputSlots) stepArgs.push(slots[s]!);
      for (const s of step.outputSlots) stepArgs.push(slots[s]!);
      await this.runAsync(step.name, ...stepArgs);
    }
    _applyReturnFixups(plan, slots);
  }

  _uniformPlanBackend(plan: ExecutionPlan): RuntimeBackend | null {
    let backend: RuntimeBackend | null = null;
    for (const step of plan.steps) {
      const entry = this._instances.get(step.name);
      if (!entry || entry.instance instanceof Promise) return null;
      if (backend === null) backend = entry.backend;
      else if (entry.backend !== backend) return null;
    }
    return backend;
  }

  static _extractShapeParams(shapeParamMap: ShapeParamMap, tensorShapes: Map<number, number[]>, args: readonly RuntimeArg[], bufferMap?: BufferMap): number[] {
    const bufferIndex = new Map<string, number>();
    if (bufferMap) {
      let i = 0;
      for (const [k, buf] of bufferMap) {
        const name = typeof k === 'string' ? k : (buf && buf.name);
        if (typeof name === 'string') bufferIndex.set(name, i);
        i++;
      }
    }
    const seen = new Map<string, true>();
    const result: number[] = [];
    for (const [key, varNode] of shapeParamMap) {
      if (seen.has(varNode.name)) continue;
      seen.set(varNode.name, true);
      const sepIdx = key.lastIndexOf(':');
      const bufferName = key.substring(0, sepIdx);
      const dimIdx = parseInt(key.substring(sepIdx + 1), 10);
      let resolved = null;
      if (bufferIndex.has(bufferName)) {
        const shape = tensorShapes.get(bufferIndex.get(bufferName)!);
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

  getKernelSource(name: string): string | null {
    const kernel = this.kernels.get(name);
    return kernel ? kernel.source : null;
  }

  getKernelSnippet(name: string): string | null {
    const kernel = this.kernels.get(name);
    return kernel ? kernel.snippet() : null;
  }

  listKernels(): string[] {
    return this.kernels.names();
  }

  serialize(): SerializedRuntimeModule {
    const entries: SerializedRuntimeModule['kernels'] = [];
    for (const name of this.kernels.names()) {
      const k = this.kernels.get(name);
      entries.push({ name: k!.name, source: k!.source, target: k!.target.name, metadata: k!.metadata });
    }
    return { name: this.name, kernels: entries };
  }

  static deserialize(data: SerializedRuntimeModule): RuntimeModule {
    const mod = new RuntimeModule(data.name);
    for (const e of data.kernels) {
      mod.addCompiledKernel(new CompiledKernel(e.name, e.source, { name: e.target }, e.metadata) as unknown as RuntimeKernel);
    }
    return mod;
  }
}
