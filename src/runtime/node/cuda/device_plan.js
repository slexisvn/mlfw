import { cu, checkCU } from './ffi.js';
import { compileToPTX, hashSource } from './program.js';
import { setDevice, devSync, acquireDevice, releaseDevice, devH2D, devD2H, devAddr } from './runtime_api.js';
import { getDevice } from './device.js';
import { beginEagerCapture, endEagerCapture, replay, syncStream } from './eager_graph.js';
import { cublasMatmulDevice } from './cublas.js';
import { scalarParam } from './launcher.js';

const _funcCache = new Map();
const _graphCache = new WeakMap();
const _residentPlans = new WeakMap();

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

function _runStepGraphed(st, dptr, stream) {
  const ordered = st.inputSlots.concat(st.outputSlots);
  const meta = st.kernel.metadata;
  if (meta.cublas) {
    const { M, N, K, aIdx, bIdx, cIdx, transB } = meta.cublas;
    cublasMatmulDevice(M, N, K, dptr[ordered[aIdx]], dptr[ordered[bIdx]], dptr[ordered[cIdx]], transB);
  } else {
    const addrs = ordered.map(s => devAddr(dptr[s]));
    launchOnPrimary(funcsFor(st), meta.gridDim, meta.blockDim, 0, addrs, st.shapeValues || [], stream);
  }
}

let _funcsCtx = null;
function funcsFor(st) { return _funcsCtx.get(st.name); }

function runCudaPlanGraphed(plan, slots, steps) {
  setDevice();
  const stream = getDevice().stream;
  const funcs = new Map();
  for (const st of steps) {
    if (!st.kernel.metadata.cublas) funcs.set(st.name, loadFunctionOnPrimary(st.kernel.source, st.kernel.name));
  }
  _funcsCtx = funcs;
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
    beginEagerCapture();
    try {
      for (const st of steps) _runStepGraphed(st, dptr, stream);
      const cap = endEagerCapture();
      g = { dptr, graph: cap.graph, exec: cap.exec };
    } catch (e) {
      try { endEagerCapture(); } catch (_) {}
      throw e;
    }
    _graphCache.set(plan, g);
  } else {
    for (let s = 0; s < plan.numSlots; s++) {
      if (slots[s] && !written.has(s)) devH2D(g.dptr[s], slots[s].data);
    }
  }
  replay(g.exec);
  syncStream();
  for (let s = 0; s < plan.numSlots; s++) {
    if (g.dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, g.dptr[s]);
  }
}

function _launchSteps(steps, dptr, funcs) {
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
}

function _runCudaPlanResident(plan, slots, steps, funcs, written) {
  let entry = _residentPlans.get(plan);
  if (!entry) {
    entry = { dptr: new Array(plan.numSlots).fill(null), sizes: new Array(plan.numSlots).fill(0), pin: new Map() };
    _residentPlans.set(plan, entry);
  }
  const { dptr, sizes, pin } = entry;
  const argSet = new Set(plan.argSlots);

  for (let s = 0; s < plan.numSlots; s++) {
    const t = slots[s];
    if (!t) continue;
    const bytes = Math.max(t.data.byteLength, 1);
    if (dptr[s] === null || sizes[s] !== bytes) {
      if (dptr[s] !== null) releaseDevice(dptr[s], sizes[s]);
      dptr[s] = acquireDevice(bytes);
      sizes[s] = bytes;
      pin.delete(s);
    }
    if (written.has(s)) continue;
    const res = t.resident;
    if (res) {
      const prev = pin.get(s);
      if (prev && prev.key === res.key && prev.version === res.version) continue;
      pin.set(s, { key: res.key, version: res.version });
    }
    devH2D(dptr[s], t.data);
  }

  _launchSteps(steps, dptr, funcs);
  devSync();

  for (let s = 0; s < plan.numSlots; s++) {
    if (dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, dptr[s]);
  }
}

export async function runCudaPlan(plan, slots, steps, opts) {
  if (_useCudaGraph) {
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

  if (opts && opts.resident) {
    _runCudaPlanResident(plan, slots, steps, funcs, written);
    return;
  }

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

  _launchSteps(steps, dptr, funcs);
  devSync();

  const argSet = new Set(plan.argSlots);
  for (let s = 0; s < plan.numSlots; s++) {
    if (dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, dptr[s]);
  }
  for (let s = 0; s < plan.numSlots; s++) {
    if (dptr[s] !== null) releaseDevice(dptr[s], sizes[s]);
  }
}
