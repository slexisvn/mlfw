import { acquire, release, copyHostToDevice, copyDeviceToHost, copyHostToDeviceAsync, copyDeviceToHostAsync } from './memory.js';
import { getDevice } from './device.js';
import { cu, checkCU } from './ffi.js';
import { setEagerDeferred, isEagerDeferred, setEagerFlushHook, isEagerCapturing, setCudaGraphArmed } from '../../../dispatcher/eager_mode.js';

export { setEagerDeferred, isEagerDeferred, setCudaGraphArmed };

const CAPACITY = 1024;
const _safe = new Map();

function _safeEntry(hostArray) {
  let e = _safe.get(hostArray);
  if (e) { _safe.delete(hostArray); _safe.set(hostArray, e); return e; }
  if (_safe.size >= CAPACITY) {
    const k = _safe.keys().next().value;
    const o = _safe.get(k);
    _safe.delete(k);
    release(o.dptr, o.bytes);
  }
  e = { dptr: acquire(hostArray.byteLength), bytes: hostArray.byteLength, valid: false };
  _safe.set(hostArray, e);
  return e;
}

export function uploadIfStale(hostArray) {
  const e = _safeEntry(hostArray);
  if (!e.valid) { copyHostToDevice(e.dptr, hostArray); e.valid = true; }
  return e.dptr;
}

export function downloadAndValidate(hostArray, dptr) {
  copyDeviceToHost(hostArray, dptr);
  const e = _safe.get(hostArray);
  if (e) e.valid = true;
}

const DEFERRED_MEM_FRACTION = 0.5;
const _def = new Map();
const _pinned = new Set();
const _capturePinned = new Set();
let _defBytes = 0;
let _defCap = 0;

export function pinResident(hostArray) { _pinned.add(hostArray); }
export function unpinResident(hostArray) { _pinned.delete(hostArray); }
export function clearCapturePins() { _capturePinned.clear(); }
export function deviceBufferDptr(hostArray) { const e = _def.get(hostArray); return e ? e.dptr : null; }
export function uploadStaticAsync(dptr, hostArray) { copyHostToDeviceAsync(dptr, hostArray); }
export function downloadStaticAsync(hostArray, dptr) { copyDeviceToHostAsync(hostArray, dptr); }

function _cap() {
  if (!_defCap) _defCap = Math.floor(getDevice().totalMem * DEFERRED_MEM_FRACTION);
  return _defCap;
}

function _syncDownload(hostArray, dptr) {
  if (isEagerCapturing()) throw new Error('illegal device sync during CUDA graph capture');
  checkCU('cuStreamSynchronize', cu.streamSynchronize(getDevice().stream));
  copyDeviceToHost(hostArray, dptr);
}

function _evict(incoming) {
  const cap = _cap();
  for (const [k, e] of _def) {
    if (_defBytes + incoming <= cap) break;
    if (_pinned.has(k) || _capturePinned.has(k)) continue;
    if (e.hostStale) _syncDownload(k, e.dptr);
    release(e.dptr, e.bytes);
    _defBytes -= e.bytes;
    _def.delete(k);
  }
}

function _defEntry(hostArray) {
  let e = _def.get(hostArray);
  if (e) { _def.delete(hostArray); _def.set(hostArray, e); }
  else {
    if (!isEagerCapturing()) _evict(hostArray.byteLength);
    e = { dptr: acquire(hostArray.byteLength), bytes: hostArray.byteLength, deviceFresh: false, hostStale: false };
    _def.set(hostArray, e);
    _defBytes += e.bytes;
  }
  if (isEagerCapturing()) _capturePinned.add(hostArray);
  return e;
}

function _upload(dptr, hostArray) {
  if (isEagerCapturing()) copyHostToDeviceAsync(dptr, hostArray);
  else copyHostToDevice(dptr, hostArray);
}

export function deviceBufferForInput(hostArray) {
  const e = _defEntry(hostArray);
  if (!e.deviceFresh) { _upload(e.dptr, hostArray); e.deviceFresh = true; }
  return e.dptr;
}

export function deviceBufferForOutput(hostArray) {
  const e = _defEntry(hostArray);
  e.deviceFresh = true;
  e.hostStale = true;
  return e.dptr;
}

export function deviceBufferForInplace(hostArray) {
  const e = _defEntry(hostArray);
  if (!e.deviceFresh) { _upload(e.dptr, hostArray); e.deviceFresh = true; }
  e.hostStale = true;
  return e.dptr;
}

export function flushDeferred() {
  for (const [k, e] of _def) {
    if (_pinned.has(k) || _capturePinned.has(k)) continue;
    release(e.dptr, e.bytes);
    _defBytes -= e.bytes;
    _def.delete(k);
  }
}

export function releaseAllResident() {
  for (const [k, e] of _def) {
    if (e.hostStale) _syncDownload(k, e.dptr);
    release(e.dptr, e.bytes);
  }
  _def.clear();
  _defBytes = 0;
  _pinned.clear();
  _capturePinned.clear();
  for (const e of _safe.values()) release(e.dptr, e.bytes);
  _safe.clear();
}

export function hostReadHook(hostArray) {
  const d = _def.get(hostArray);
  if (isEagerCapturing()) {
    if (d && d.hostStale) throw new Error('illegal host read of device-resident tensor during CUDA graph capture');
    return;
  }
  const s = _safe.get(hostArray);
  if (s) s.valid = false;
  if (d) {
    if (d.hostStale) { _syncDownload(hostArray, d.dptr); d.hostStale = false; }
    d.deviceFresh = false;
  }
}

setEagerFlushHook(flushDeferred);
setEagerDeferred(true);
