import { compileToPTX, getProgram } from './program.js';
import { setDevice, devSync, devH2D, devD2H } from './runtime_api.js';
import { acquire, release } from './memory.js';
import { getDevice } from './device.js';
import { beginEagerCapture, endEagerCapture, destroyEagerGraph, replay, syncStream } from './eager_graph.js';
import { cublasMatmulDevice } from './cublas.js';
import { scalarParam, devicePtrParam } from './launcher.js';
import { cu, checkCU } from './ffi.js';

const _planState = new WeakMap();
const _liveStates = new Set();
const _stateReclaim = new FinalizationRegistry((state) => _releaseState(state));

let _useCudaGraph = false;
export function setCudaGraphEnabled(on) { _useCudaGraph = on; }
export function isCudaGraphEnabled() { return _useCudaGraph; }

function _groupsOf(plan) {
  const buffers = plan.buffers;
  const count = buffers ? buffers.bufferBytes.length : plan.numSlots;
  const of = new Array(plan.numSlots);
  for (let s = 0; s < plan.numSlots; s++) of[s] = buffers ? buffers.slotBuffer[s] : s;
  return { of, count };
}

function _groupSizes(plan, slots, groups) {
  const sizes = new Array(groups.count).fill(0);
  for (let s = 0; s < plan.numSlots; s++) {
    if (!slots[s]) continue;
    const bytes = Math.max(slots[s].data.byteLength, 1);
    const g = groups.of[s];
    if (bytes > sizes[g]) sizes[g] = bytes;
  }
  return sizes;
}

function _releaseState(state) {
  if (state.released) return;
  state.released = true;
  _liveStates.delete(state);
  destroyEagerGraph(state);
  state.graph = null;
  state.exec = null;
  for (let g = 0; g < state.gptr.length; g++) {
    if (state.gptr[g] === null) continue;
    release(state.gptr[g], state.sizes[g]);
    state.gptr[g] = null;
  }
  state.dptr.fill(null);
}

function _stateFor(plan, groups) {
  let state = _planState.get(plan);
  if (state) return state;
  state = {
    dptr: new Array(plan.numSlots).fill(null),
    gptr: new Array(groups.count).fill(null),
    sizes: new Array(groups.count).fill(0),
    pin: new Map(),
    graph: null,
    exec: null,
    released: false,
  };
  _planState.set(plan, state);
  _liveStates.add(state);
  _stateReclaim.register(plan, state, plan);
  return state;
}

function _dropState(plan) {
  const state = _planState.get(plan);
  if (!state) return;
  _stateReclaim.unregister(plan);
  _planState.delete(plan);
  _releaseState(state);
}

export function releaseAllPlanBuffers() {
  for (const state of [..._liveStates]) _releaseState(state);
  _liveStates.clear();
}

function launchOnPrimary(func, gridDim, blockDim, sharedMemBytes, deviceAddrs, scalars, stream = null) {
  const params = [];
  for (const a of deviceAddrs) params.push(devicePtrParam(a));
  for (const s of scalars) params.push(scalarParam(s));
  checkCU('cuLaunchKernel', cu.launchKernel(
    func,
    gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2],
    sharedMemBytes, stream, params, null,
  ));
}

function _launchSteps(steps, dptr, funcs, stream = null) {
  for (const st of steps) {
    const ordered = st.inputSlots.concat(st.outputSlots);
    const meta = st.kernel.metadata;
    if (meta.cublas) {
      const { M, N, K, aIdx, bIdx, cIdx, transB } = meta.cublas;
      cublasMatmulDevice(M, N, K, dptr[ordered[aIdx]], dptr[ordered[bIdx]], dptr[ordered[cIdx]], transB);
    } else {
      launchOnPrimary(funcs.get(st.name), meta.gridDim, meta.blockDim, 0, ordered.map((s) => dptr[s]), st.shapeValues || [], stream);
    }
  }
}

function _loadFuncs(steps) {
  setDevice();
  const funcs = new Map();
  for (const st of steps) {
    if (st.kernel.metadata.cublas) continue;
    funcs.set(st.name, getProgram(st.kernel.source, st.kernel.name).func);
  }
  return funcs;
}

function _writtenSlots(steps) {
  const written = new Set();
  for (const st of steps) for (const s of st.outputSlots) written.add(s);
  return written;
}

function runCudaPlanGraphed(plan, slots, steps) {
  const funcs = _loadFuncs(steps);
  const stream = getDevice().stream;
  const written = _writtenSlots(steps);
  const argSet = new Set(plan.argSlots);
  const groups = _groupsOf(plan);

  const state = _stateFor(plan, groups);
  if (!state.exec) {
    const sizes = _groupSizes(plan, slots, groups);
    for (let g = 0; g < groups.count; g++) {
      if (sizes[g] === 0) continue;
      state.gptr[g] = acquire(sizes[g]);
      state.sizes[g] = sizes[g];
    }
    for (let s = 0; s < plan.numSlots; s++) {
      if (!slots[s]) continue;
      state.dptr[s] = state.gptr[groups.of[s]];
      if (!written.has(s)) devH2D(state.dptr[s], slots[s].data);
    }
    beginEagerCapture();
    try {
      _launchSteps(steps, state.dptr, funcs, stream);
      const cap = endEagerCapture();
      state.graph = cap.graph;
      state.exec = cap.exec;
    } catch (e) {
      try { endEagerCapture(); } catch (_) {}
      _dropState(plan);
      throw e;
    }
  } else {
    for (let s = 0; s < plan.numSlots; s++) {
      if (slots[s] && !written.has(s)) devH2D(state.dptr[s], slots[s].data);
    }
  }
  replay(state.exec);
  syncStream();
  for (let s = 0; s < plan.numSlots; s++) {
    if (state.dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, state.dptr[s]);
  }
}

function runCudaPlanResident(plan, slots, steps, funcs, written) {
  const groups = _groupsOf(plan);
  const state = _stateFor(plan, groups);
  const { dptr, gptr, sizes, pin } = state;
  const argSet = new Set(plan.argSlots);
  const wanted = _groupSizes(plan, slots, groups);

  const moved = new Set();
  for (let g = 0; g < groups.count; g++) {
    if (wanted[g] === 0 || (gptr[g] !== null && sizes[g] === wanted[g])) continue;
    if (gptr[g] !== null) release(gptr[g], sizes[g]);
    gptr[g] = acquire(wanted[g]);
    sizes[g] = wanted[g];
    moved.add(g);
  }

  for (let s = 0; s < plan.numSlots; s++) {
    const t = slots[s];
    if (!t) continue;
    const g = groups.of[s];
    if (moved.has(g)) pin.delete(s);
    dptr[s] = gptr[g];
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

function runCudaPlanTransient(plan, slots, steps, funcs, written) {
  const groups = _groupsOf(plan);
  const sizes = _groupSizes(plan, slots, groups);
  const gptr = new Array(groups.count).fill(null);
  const dptr = new Array(plan.numSlots).fill(null);
  try {
    for (let g = 0; g < groups.count; g++) {
      if (sizes[g] > 0) gptr[g] = acquire(sizes[g]);
    }
    for (let s = 0; s < plan.numSlots; s++) {
      if (!slots[s]) continue;
      dptr[s] = gptr[groups.of[s]];
      if (!written.has(s)) devH2D(dptr[s], slots[s].data);
    }
    _launchSteps(steps, dptr, funcs);
    devSync();
    const argSet = new Set(plan.argSlots);
    for (let s = 0; s < plan.numSlots; s++) {
      if (dptr[s] !== null && written.has(s) && argSet.has(s)) devD2H(slots[s].data, dptr[s]);
    }
  } finally {
    for (let g = 0; g < groups.count; g++) {
      if (gptr[g] !== null) release(gptr[g], sizes[g]);
    }
  }
}

export async function runCudaPlan(plan, slots, steps, opts) {
  if (_useCudaGraph) return runCudaPlanGraphed(plan, slots, steps);

  for (const st of steps) {
    if (!st.kernel.metadata.cublas) compileToPTX(st.kernel.source, st.kernel.name);
  }
  const funcs = _loadFuncs(steps);
  const written = _writtenSlots(steps);

  if (opts && opts.resident) runCudaPlanResident(plan, slots, steps, funcs, written);
  else runCudaPlanTransient(plan, slots, steps, funcs, written);
}
