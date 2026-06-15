import { encodeWat } from '../../backend/wasm/wat_encoder.js';

const _registry = new Map();

export function registerBackend(kind, backend) {
  _registry.set(kind, backend);
}

export function getBackend(kind) {
  return _registry.get(kind) || null;
}

export function hasBackend(kind) {
  return _registry.has(kind);
}

let _webgpuMod = null;
function getWebGPURuntime() {
  if (!_webgpuMod) _webgpuMod = import('./webgpu_runtime.js');
  return _webgpuMod;
}

let _wasmPoolMod = null;
function getWasmPool() {
  if (!_wasmPoolMod) _wasmPoolMod = import('#io/wasm_pool');
  return _wasmPoolMod;
}

let _cudaMod = null;
function getCudaRuntime() {
  if (!_cudaMod) _cudaMod = import('#io/cuda_runtime');
  return _cudaMod;
}

let _cudaSyncMod = null;
export async function preloadCudaRuntime() {
  if (_cudaSyncMod) return _cudaSyncMod;
  const mod = await getCudaRuntime();
  if (mod.preloadCublas) await mod.preloadCublas();
  _cudaSyncMod = mod;
  return mod;
}

const MATH_IMPORTS = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos,
  tan: Math.tan, tanh: Math.tanh, pow: Math.pow, fmod: (a, b) => a % b,
  rsqrt: x => 1 / Math.sqrt(x), sign: Math.sign, round: Math.round,
};

function instantiateWasm(kernel) {
  let binary;
  try { binary = encodeWat(kernel.source); } catch (e) { throw new Error('encodeWat: ' + e.message + '\n' + kernel.source); }
  let mod;
  try { mod = new WebAssembly.Module(binary); } catch (e) { throw new Error('WASM: ' + e.message + '\n' + kernel.source); }
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
    parallel: kernel.metadata.parallel || null,
    mathNames: kernel.metadata.imports ? [...kernel.metadata.imports.keys()] : [],
  };
}

function runWasmKernel(wasmInstance, tensorArgs, shapeValues, parStart, parEnd) {
  const { exports, memory, bufferOffsets, funcName } = wasmInstance;
  const fn = exports[funcName];
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

registerBackend('js', {
  instantiate(kernel) {
    return new Function('return ' + kernel.source)();
  },
  runSync(fn, tensorArgs, shapeValues) {
    const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
    return fn(...callArgs);
  },
  runAsync(fn, tensorArgs, shapeValues) {
    const callArgs = shapeValues ? [...tensorArgs, ...shapeValues] : tensorArgs;
    return fn(...callArgs);
  },
  isAsync() { return false; },
});

registerBackend('wasm', {
  instantiate(kernel) {
    return instantiateWasm(kernel);
  },
  runSync(inst, tensorArgs, shapeValues) {
    if (inst.parallel) runWasmKernel(inst, tensorArgs, shapeValues, 0, inst.parallel.extent);
    else runWasmKernel(inst, tensorArgs, shapeValues);
  },
  async runAsync(inst, tensorArgs, shapeValues) {
    if (inst.parallel && inst.parallel.poolSafe) {
      const { runWasmParallel } = await getWasmPool();
      await runWasmParallel(inst, inst.funcName, tensorArgs, shapeValues, inst.parallel, inst.mathNames);
      return;
    }
    this.runSync(inst, tensorArgs, shapeValues);
  },
  isAsync(inst) {
    return !!(inst && inst.parallel && inst.parallel.poolSafe);
  },
});

registerBackend('webgpu', {
  instantiate(kernel) {
    return getWebGPURuntime().then(m => m.instantiateWebGPU(kernel));
  },
  runSync() {
    throw new Error('WebGPU kernel requires async execution — use runAsync()');
  },
  async runAsync(inst, tensorArgs, shapeValues) {
    const { runWebGPUKernel } = await getWebGPURuntime();
    await runWebGPUKernel(inst, tensorArgs, shapeValues);
  },
  isAsync() { return true; },
});

registerBackend('cuda', {
  instantiate(kernel) {
    return { kernel };
  },
  runSync(inst, tensorArgs, shapeValues) {
    if (!_cudaSyncMod) throw new Error('CUDA sync runtime not preloaded — call preloadCudaRuntime() before synchronous execution');
    _cudaSyncMod.runCudaKernelResident(inst.kernel, tensorArgs, shapeValues);
  },
  async runAsync(inst, tensorArgs, shapeValues) {
    const { runCudaKernel } = await getCudaRuntime();
    await runCudaKernel(inst.kernel, tensorArgs, shapeValues);
  },
  async runPlan(plan, slots, steps) {
    const { runCudaPlan } = await getCudaRuntime();
    await runCudaPlan(plan, slots, steps);
  },
  isAsync() { return true; },
});
