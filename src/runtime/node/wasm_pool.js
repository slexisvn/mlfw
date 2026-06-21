import { Worker } from 'node:worker_threads';
import os from 'node:os';

const WORKER_SRC = `
const { parentPort } = require('node:worker_threads');

const MATH = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos,
  tan: Math.tan, tanh: Math.tanh, pow: Math.pow,
  fmod: (a, b) => a % b,
  rsqrt: x => 1 / Math.sqrt(x),
  sign: Math.sign, round: Math.round,
};

parentPort.on('message', (msg) => {
  const { wasmBinary, mathImportNames, bufferEntries,
          callArgs, controlBuffer, workerIdx, outputIndices } = msg;

  const compiled = new WebAssembly.Module(new Uint8Array(wasmBinary));

  const mathImports = {};
  for (const name of mathImportNames) {
    mathImports[name] = MATH[name] || Math[name] || (x => x);
  }

  const instance = new WebAssembly.Instance(compiled, { math: mathImports });
  const wasmMem = instance.exports.memory;
  const fn = Object.values(instance.exports).find(v => typeof v === 'function');

  for (const entry of bufferEntries) {
    new Float32Array(wasmMem.buffer, entry.offset, entry.length).set(new Float32Array(entry.data));
  }

  fn(...callArgs);

  const outSet = new Set(outputIndices);
  const outputs = [];
  const transferable = [];
  for (let i = 0; i < bufferEntries.length; i++) {
    if (outSet.has(i)) {
      const entry = bufferEntries[i];
      const out = new Float32Array(entry.length);
      out.set(new Float32Array(wasmMem.buffer, entry.offset, entry.length));
      outputs.push(out.buffer);
      transferable.push(out.buffer);
    } else {
      outputs.push(null);
    }
  }
  const ctrl = new Int32Array(controlBuffer);
  parentPort.postMessage({ workerIdx, outputs }, transferable);
  Atomics.store(ctrl, workerIdx, 1);
  Atomics.notify(ctrl, workerIdx);
});
`;

let _pool = null;

function getPool() {
  if (!_pool) {
    const numWorkers = Math.max(1, Math.min(os.cpus().length - 1, 16));
    _pool = new WasmThreadPool(numWorkers);
  }
  return _pool;
}

class WasmThreadPool {
  constructor(numWorkers) {
    this.numWorkers = numWorkers;
    this._workers = [];
    this._ready = [];
  }

  _ensureWorkers() {
    if (this._workers.length > 0) return;
    for (let i = 0; i < this.numWorkers; i++) {
      const w = new Worker(WORKER_SRC, { eval: true });
      w.unref();
      this._workers.push(w);
    }
  }

  dispatch(workerIdx, msg) {
    this._ensureWorkers();
    return new Promise((resolve) => {
      const handler = (result) => {
        if (result.workerIdx === workerIdx) {
          this._workers[workerIdx].removeListener('message', handler);
          resolve(result);
        }
      };
      this._workers[workerIdx].on('message', handler);
      this._workers[workerIdx].postMessage(msg);
    });
  }

  terminate() {
    for (const w of this._workers) w.terminate();
    this._workers = [];
  }
}

export async function runWasmParallel(wasmInst, name, tensorArgs, shapeValues, parallel, mathImportNames) {
  const pool = getPool();
  pool._ensureWorkers();

  const { binary, bufferOffsets } = wasmInst;
  const offsets = [...bufferOffsets.values()];
  const nBufs = Math.min(offsets.length, tensorArgs.length);
  const { extent, outputIndices } = parallel;
  const outSet = new Set(outputIndices || []);

  const activeWorkers = Math.min(pool.numWorkers, Math.max(1, extent));
  const chunkSize = Math.ceil(extent / activeWorkers);

  const strides = [];
  for (let i = 0; i < nBufs; i++) {
    const data = tensorArgs[i];
    const bufLen = data instanceof Float32Array ? data.length : 0;
    const s = bufLen / extent;
    strides.push(s >= 1 && Number.isInteger(s) ? s : 0);
  }

  const baseCallArgs = offsets.slice(0, nBufs);
  if (shapeValues) {
    for (const v of shapeValues) baseCallArgs.push(v);
  }

  const controlBuffer = new SharedArrayBuffer(4 * activeWorkers);
  const wasmBinary = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);

  const promises = [];
  for (let w = 0; w < activeWorkers; w++) {
    const parStart = w * chunkSize;
    const parEnd = Math.min(parStart + chunkSize, extent);
    if (parStart >= extent) break;

    const bufferEntries = [];
    for (let i = 0; i < nBufs; i++) {
      const data = tensorArgs[i];
      if (!(data instanceof Float32Array)) continue;
      bufferEntries.push({
        offset: offsets[i],
        length: data.length,
        data: data.buffer.slice(0),
        fullLength: data.length,
        elemStart: 0,
      });
    }

    const callArgs = [...baseCallArgs, parStart, parEnd];

    promises.push(pool.dispatch(w, {
      wasmBinary,
      mathImportNames,
      bufferEntries,
      callArgs,
      controlBuffer,
      workerIdx: w,
      outputIndices: outputIndices || [],
    }));
  }

  const results = await Promise.all(promises);

  for (let i = 0; i < nBufs; i++) {
    if (!outSet.has(i) && outSet.size > 0) continue;
    const data = tensorArgs[i];
    if (!(data instanceof Float32Array)) continue;
    const stride = strides[i];

    if (stride > 0) {
      for (const r of results) {
        const outBuf = r.outputs[i];
        if (!outBuf) continue;
        const wIdx = r.workerIdx;
        const parStart = wIdx * chunkSize;
        const parEnd = Math.min(parStart + chunkSize, extent);
        const workerFull = new Float32Array(outBuf);
        const elemStart = parStart * stride;
        const elemEnd = parEnd * stride;
        data.set(workerFull.subarray(elemStart, elemEnd), elemStart);
      }
    } else {
      const firstResult = results.find(r => r.workerIdx === 0);
      if (firstResult && firstResult.outputs[i]) {
        data.set(new Float32Array(firstResult.outputs[i]));
      }
    }
  }
}
