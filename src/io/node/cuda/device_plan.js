import { cu, checkCU } from './ffi.js';
import { compileToPTX, hashSource } from './program.js';
import { setDevice, devSync, acquireDevice, releaseDevice, devH2D, devD2H, devAddr } from './runtime_api.js';
import { cublasMatmulDevice } from './cublas.js';

const _funcCache = new Map();

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

function launchOnPrimary(func, gridDim, blockDim, sharedMemBytes, deviceAddrs, scalars) {
  const params = [];
  for (const a of deviceAddrs) params.push(bufferParam(a));
  for (const s of scalars) params.push(scalarParam(s));
  checkCU('cuLaunchKernel', cu.launchKernel(
    func,
    gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2],
    sharedMemBytes, null, params, null,
  ));
}

export async function runCudaPlan(plan, slots, steps) {
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
