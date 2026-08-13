import { encodeWat } from '../backend/wasm/wat_encoder.js';
import { erfScalar, erfcScalar, lgammaScalar, gammaScalar, digammaScalar } from '../util/special_math.js';
import { registerMeasurer } from './measurer_registry.js';
import type { Measurer } from './measurer_registry.js';
import type { NumericTypedArray } from '../tensor/types/dtype.js';
import type { ExecutionPlan, PlanStepRuntime, RuntimeKernel, RuntimeTensor } from './runtime.js';
import type { CudaKernel } from './io.js';
import type { WebGPUInstance, WebGPUKernel } from './webgpu.js';
import type { DynamicFn } from '../dispatcher/jit_dispatch.js';
import type { WasmBufferView, WasmBufferViewCtor, WasmInstance, WasmKernelFn } from './io.js';

type BackendInstance = unknown;
type JsKernelFn = (...args: (NumericTypedArray | number)[]) => unknown;
type Backend = {
  instantiate(kernel: RuntimeKernel): BackendInstance;
  runSync(instance: BackendInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): unknown;
  runAsync(instance: BackendInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): Promise<unknown> | unknown;
  runPlan?: (plan: ExecutionPlan, slots: Array<RuntimeTensor | null>, steps: PlanStepRuntime[], opts?: Record<string, unknown>) => Promise<unknown>;
  isAsync(instance?: BackendInstance, kernel?: RuntimeKernel): boolean;
};
type CudaBackendInstance = { kernel: RuntimeKernel };
type MathImportFn = (...args: number[]) => number;
type WasmPoolModule = typeof import('#io/wasm_pool');
type CudaRuntimeModule = typeof import('#io/cuda_runtime');
type WebGPUModule = typeof import('./webgpu.js');
type MeasureOpts = { warmup?: number; repeat?: number };

const _registry = new Map<string, Backend>();

export function registerBackend(kind: string, backend: Backend): void {
  _registry.set(kind, backend);
}

export function getBackend(kind: string): Backend | null {
  return _registry.get(kind) || null;
}

export function hasBackend(kind: string): boolean {
  return _registry.has(kind);
}

let _wasmPoolMod: Promise<WasmPoolModule> | null = null;
function getWasmPool(): Promise<WasmPoolModule> {
  if (!_wasmPoolMod) _wasmPoolMod = import('#io/wasm_pool');
  return _wasmPoolMod;
}

let _cudaMod: Promise<CudaRuntimeModule> | null = null;
function getCudaRuntime(): Promise<CudaRuntimeModule> {
  if (!_cudaMod) _cudaMod = import('#io/cuda_runtime');
  return _cudaMod;
}

let _cudaSyncMod: CudaRuntimeModule | null = null;
export async function preloadCudaRuntime(): Promise<CudaRuntimeModule> {
  if (_cudaSyncMod) return _cudaSyncMod;
  const mod = await getCudaRuntime();
  if (mod.preloadCublas) await mod.preloadCublas();
  _cudaSyncMod = mod;
  return mod;
}

export async function releaseCudaMemory(): Promise<number> {
  if (!_cudaMod) return 0;
  const mod = await _cudaMod;
  return mod.releaseCudaMemory();
}

let _webgpuMod: WebGPUModule | null = null;
export async function preloadWebGPU(): Promise<WebGPUModule> {
  if (_webgpuMod) return _webgpuMod;
  const mod = await import('./webgpu.js');
  const { setWebGPUEagerFn, setWebgpuRNN } = await import('../dispatcher/jit_dispatch.js');
  await mod.ensureWebGPUEager();
  setWebGPUEagerFn(mod.webgpuEagerOp as DynamicFn);
  setWebgpuRNN(mod.webgpuRNN as DynamicFn);
  _webgpuMod = mod;
  return mod;
}

const MATH_IMPORTS: Record<string, MathImportFn> = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos,
  tan: Math.tan, tanh: Math.tanh, pow: Math.pow, fmod: (a: number, b: number) => a % b,
  rsqrt: (x: number) => 1 / Math.sqrt(x), sign: Math.sign, round: Math.round,
  erf: erfScalar, erfc: erfcScalar, lgamma: lgammaScalar, gamma: gammaScalar, digamma: digammaScalar,
};

function instantiateWasm(kernel: RuntimeKernel): WasmInstance {
  let binary: Uint8Array<ArrayBuffer>;
  try { binary = encodeWat(kernel.source); } catch (e) { throw new Error('encodeWat: ' + (e as Error).message + '\n' + kernel.source); }
  let mod: WebAssembly.Module;
  try { mod = new WebAssembly.Module(binary); } catch (e) { throw new Error('WASM: ' + (e as Error).message + '\n' + kernel.source); }
  const mathImports: Record<string, MathImportFn> = {};
  if (kernel.metadata.imports) {
    for (const [name] of kernel.metadata.imports) {
      mathImports[name] = MATH_IMPORTS[name] || (Math as unknown as Record<string, MathImportFn>)[name] || ((x: number) => x);
    }
  }
  const instance = new WebAssembly.Instance(mod, { math: mathImports });
  return {
    exports: instance.exports,
    memory: instance.exports.memory as WebAssembly.Memory,
    bufferOffsets: kernel.metadata.bufferOffsets as Map<string, number>,
    funcName: kernel.name,
    binary,
    parallel: (kernel.metadata.parallel as WasmInstance['parallel']) || null,
    mathNames: kernel.metadata.imports ? [...kernel.metadata.imports.keys()] : [],
  };
}

function runWasmKernel(wasmInstance: WasmInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null, parStart?: number, parEnd?: number): void {
  const { exports, memory, bufferOffsets, funcName } = wasmInstance;
  const fn = exports[funcName] as WasmKernelFn;
  const offsets = [...bufferOffsets.values()];
  const nBufs = Math.min(offsets.length, tensorArgs.length);

  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    if (ArrayBuffer.isView(data)) {
      new (data.constructor as WasmBufferViewCtor)(memory.buffer, offsets[i], data.length).set(data);
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
      (data as WasmBufferView).set(new (data.constructor as WasmBufferViewCtor)(memory.buffer, offsets[i], data.length));
    }
  }
}

registerBackend('js', {
  instantiate(kernel: RuntimeKernel): JsKernelFn {
    return new Function('return ' + kernel.source)() as JsKernelFn;
  },
  runSync(fn: JsKernelFn, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): unknown {
    const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
    return fn(...callArgs);
  },
  runAsync(fn: JsKernelFn, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): unknown {
    const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
    return fn(...callArgs);
  },
  isAsync(): boolean { return false; },
});

registerBackend('wasm', {
  instantiate(kernel: RuntimeKernel): WasmInstance {
    return instantiateWasm(kernel);
  },
  runSync(inst: WasmInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): void {
    if (inst.parallel) runWasmKernel(inst, tensorArgs, shapeValues, 0, inst.parallel.extent);
    else runWasmKernel(inst, tensorArgs, shapeValues);
  },
  async runAsync(inst: WasmInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): Promise<void> {
    if (inst.parallel && inst.parallel.poolSafe) {
      const { runWasmParallel } = await getWasmPool();
      await runWasmParallel(inst, inst.funcName, tensorArgs, shapeValues, inst.parallel, inst.mathNames);
      return;
    }
    this.runSync(inst, tensorArgs, shapeValues);
  },
  isAsync(inst?: WasmInstance, kernel?: RuntimeKernel): boolean {
    const parallel = inst ? inst.parallel : kernel && kernel.metadata && kernel.metadata.parallel;
    return !!(parallel && parallel.poolSafe);
  },
});

function measureWasm(compiled: RuntimeKernel, byteSizes: readonly number[], shapeValues: readonly number[] = [], opts: MeasureOpts = {}): number[] {
  const warmup = opts.warmup ?? 5;
  const repeat = opts.repeat ?? 30;
  const inst = instantiateWasm(compiled);
  const tensorArgs = byteSizes.map((bytes) => new Float32Array(Math.max(1, Math.ceil(bytes / 4))));
  const sv = shapeValues || [];
  const once = () => {
    if (inst.parallel) runWasmKernel(inst, tensorArgs, sv, 0, inst.parallel.extent);
    else runWasmKernel(inst, tensorArgs, sv);
  };
  for (let i = 0; i < warmup; i++) once();
  const samples: number[] = [];
  for (let i = 0; i < repeat; i++) {
    const t0 = performance.now();
    once();
    samples.push(performance.now() - t0);
  }
  return samples;
}

registerMeasurer('wasm', measureWasm as Measurer);

registerBackend('webgpu', {
  instantiate(kernel: RuntimeKernel): Promise<unknown> {
    return preloadWebGPU().then(m => m.instantiateWebGPU(kernel as unknown as WebGPUKernel));
  },
  runSync(): never {
    throw new Error('WebGPU kernel requires async execution — use runAsync()');
  },
  async runAsync(inst: unknown, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): Promise<void> {
    const { runWebGPUKernel } = await preloadWebGPU();
    await runWebGPUKernel(inst as WebGPUInstance, tensorArgs, shapeValues);
  },
  async runPlan(plan: ExecutionPlan, slots: Array<RuntimeTensor | null>, steps: PlanStepRuntime[]): Promise<void> {
    const { runWebGPUPlan } = await preloadWebGPU();
    await runWebGPUPlan(plan, slots, steps);
  },
  isAsync(): boolean { return true; },
});

registerBackend('cuda', {
  instantiate(kernel: RuntimeKernel): CudaBackendInstance {
    return { kernel };
  },
  runSync(inst: CudaBackendInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): void {
    if (!_cudaSyncMod) throw new Error('CUDA sync runtime not preloaded — call preloadCudaRuntime() before synchronous execution');
    _cudaSyncMod.runCudaKernelResident(inst.kernel as unknown as CudaKernel, tensorArgs, shapeValues);
  },
  async runAsync(inst: CudaBackendInstance, tensorArgs: readonly NumericTypedArray[], shapeValues: readonly number[] | null): Promise<void> {
    const { runCudaKernel } = await getCudaRuntime();
    await runCudaKernel(inst.kernel as unknown as CudaKernel, tensorArgs, shapeValues);
  },
  async runPlan(plan: ExecutionPlan, slots: Array<RuntimeTensor | null>, steps: PlanStepRuntime[], opts?: Record<string, unknown>): Promise<void> {
    const { runCudaPlan } = await getCudaRuntime();
    await runCudaPlan(plan, slots, steps, opts);
  },
  isAsync(): boolean { return true; },
});
