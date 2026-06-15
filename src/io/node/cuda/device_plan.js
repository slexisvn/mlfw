import { cu, checkCU } from './ffi.js';
import { compileToPTX, hashSource } from './program.js';
import { setDevice, devSync, acquireDevice, releaseDevice, devH2D, devD2H, devAddr,
         getCaptureStream, beginCapture, endCaptureInstantiate, graphLaunch, streamSync } from './runtime_api.js';
import { cublasMatmulDevice } from './cublas.js';

const _funcCache = new Map();
const _graphCache = new WeakMap();

let _useCudaGraph = false;
export function setCudaGraphEnabled(on) { _useCudaGraph = on; }
export function isCudaGraphEnabled() { return _useCudaGraph; }

function loadFunctionOnPrimary(source, kernelName) {
  const key = hashSource(source) + ':' + kernelName;
  const cached = _funcCache.get(key);
  if (cached) return cached;
  const ptx = compileToPTX(source, kernelName);
  setDevice();
  const mod = [null];
  checkCU('cuModuleLoadData', cu.moduleLoadData(mod, ptx));
  const func = [null];
  checkCU('cuModuleGetFunction', cu.moduleGetFunction(func, mod[0], kernelName));
  _funcCache.set(key, func[0]);
  return func[0];
}

function bufferParam(addr) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(addr);
  return b;
}

function scalarParam(value) {
  const b = Buffer.alloc(4);
  if (Number.isInteger(value)) b.writeInt32LE(value | 0);
  else b.writeFloatLE(value);
  return b;
}

function launchOnPrimary(func, gridDim, blockDim, sharedMemBytes, deviceAddrs, scalars, stream = null) {
  const params = [];
  for (const a of deviceAddrs) params.push(bufferParam(a));
  for (const s of scalars) params.push(scalarParam(s));
  checkCU('cuLaunchKernel', cu.launchKernel(
    func,
    gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2],
    sharedMemBytes, stream, params, null,
  ));
}

function _runStepNative(st, dptr, stream) {
  const ordered = st.inputSlots.concat(st.outputSlots);
  const meta = st.kernel.metadata;
  const addrs = ordered.map(s => devAddr(dptr[s]));
  launchOnPrimary(funcsFor(st), meta.gridDim, meta.blockDim, 0, addrs, st.shapeValues || [], stream);
}

let _funcsCtx = null;
function funcsFor(st) { return _funcsCtx.get(st.name); }

function runCudaPlanGraphed(plan, slots, steps) {
  setDevice();
  const funcs = new Map();
  for (const st of steps) funcs.set(st.name, loadFunctionOnPrimary(st.kernel.source, st.kernel.name));
  _funcsCtx = funcs;
  const stream = getCaptureStream();
  const written = new Set();
  for (const st of steps) for (const s of st.outputSlots) written.add(s);
  const argSet = new Set(plan.argSlots);

  let g = _graphCache.get(plan);
  if (!g) {
    const dptr = new Array(plan.numSlots).fill(null);
    for (let s = 0; s < plan.numSlots; s++) {
      const t = slots[s];
      if (!t) continue;
      dptr[s] = acquireDevice(Math.max(t.data.byteLength, 1));
      if (!written.has(s)) devH2D(dptr[s], t.data);
    }
    beginCapture(stream);
    for (const st of steps) _runStepNative(st, dptr, stream);
    const exec = endCaptureInstantiate(stream);
    g = { dptr, exec };
    _graphCache.set(plan, g);
  } else {
    for (let s = 0; s < plan.numSlots; s++) {
      if (slots[s] && !written.has(s)) devH2D(g.dptr[s], slots[s].data);
    }
  }
  graphLaunch(g.exec, stream);
  streamSync(stream);
  for (let s = 0; s < plan.numSlots; s++) {
    if (g.dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, g.dptr[s]);
  }
}

export async function runCudaPlan(plan, slots, steps) {
  if (_useCudaGraph && !steps.some(st => st.kernel.metadata.cublas)) {
    return runCudaPlanGraphed(plan, slots, steps);
  }
  for (const st of steps) {
    if (!st.kernel.metadata.cublas) compileToPTX(st.kernel.source, st.kernel.name);
  }
  setDevice();
  const funcs = new Map();
  for (const st of steps) {
    if (!st.kernel.metadata.cublas) funcs.set(st.name, loadFunctionOnPrimary(st.kernel.source, st.kernel.name));
  }

  const written = new Set();
  for (const st of steps) for (const s of st.outputSlots) written.add(s);

  const dptr = new Array(plan.numSlots).fill(null);
  const sizes = new Array(plan.numSlots).fill(0);
  for (let s = 0; s < plan.numSlots; s++) {
    const t = slots[s];
    if (!t) continue;
    const bytes = Math.max(t.data.byteLength, 1);
    sizes[s] = bytes;
    dptr[s] = acquireDevice(bytes);
    if (!written.has(s)) devH2D(dptr[s], t.data);
  }

  for (const st of steps) {
    const ordered = st.inputSlots.concat(st.outputSlots);
    const meta = st.kernel.metadata;
    if (meta.cublas) {
      const { M, N, K, aIdx, bIdx, cIdx, transB } = meta.cublas;
      cublasMatmulDevice(M, N, K, dptr[ordered[aIdx]], dptr[ordered[bIdx]], dptr[ordered[cIdx]], transB);
    } else {
      const addrs = ordered.map(s => devAddr(dptr[s]));
      launchOnPrimary(funcs.get(st.name), meta.gridDim, meta.blockDim, 0, addrs, st.shapeValues || []);
    }
  }
  devSync();

  const argSet = new Set(plan.argSlots);
  for (let s = 0; s < plan.numSlots; s++) {
    if (dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, dptr[s]);
  }
  for (let s = 0; s < plan.numSlots; s++) {
    if (dptr[s] !== null) releaseDevice(dptr[s], sizes[s]);
  }
}
