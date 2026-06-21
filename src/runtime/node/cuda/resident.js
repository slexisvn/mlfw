import { acquire, release, copyHostToDevice, copyDeviceToHost } from './memory.js';
import { getDevice } from './device.js';
import { cu, checkCU } from './ffi.js';
import { setEagerDeferred, isEagerDeferred, setEagerFlushHook } from '../../../dispatcher/eager_mode.js';

export { setEagerDeferred, isEagerDeferred };

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
let _defBytes = 0;
let _defCap = 0;

export function pinResident(hostArray) { _pinned.add(hostArray); }
export function unpinResident(hostArray) { _pinned.delete(hostArray); }

function _cap() {
  if (!_defCap) _defCap = Math.floor(getDevice().totalMem * DEFERRED_MEM_FRACTION);
  return _defCap;
}

function _syncDownload(hostArray, dptr) {
  checkCU('cuStreamSynchronize', cu.streamSynchronize(getDevice().stream));
  copyDeviceToHost(hostArray, dptr);
}

function _evict(incoming) {
  const cap = _cap();
  for (const [k, e] of _def) {
    if (_defBytes + incoming <= cap) break;
    if (_pinned.has(k)) continue;
    if (e.hostStale) _syncDownload(k, e.dptr);
    release(e.dptr, e.bytes);
    _defBytes -= e.bytes;
    _def.delete(k);
  }
}

function _defEntry(hostArray) {
  let e = _def.get(hostArray);
  if (e) { _def.delete(hostArray); _def.set(hostArray, e); return e; }
  _evict(hostArray.byteLength);
  e = { dptr: acquire(hostArray.byteLength), bytes: hostArray.byteLength, deviceFresh: false, hostStale: false };
  _def.set(hostArray, e);
  _defBytes += e.bytes;
  return e;
}

export function deviceBufferForInput(hostArray) {
  const e = _defEntry(hostArray);
  if (!e.deviceFresh) { copyHostToDevice(e.dptr, hostArray); e.deviceFresh = true; }
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
  if (!e.deviceFresh) { copyHostToDevice(e.dptr, hostArray); e.deviceFresh = true; }
  e.hostStale = true;
  return e.dptr;
}

export function flushDeferred() {
  for (const [k, e] of _def) {
    if (_pinned.has(k)) continue;
    release(e.dptr, e.bytes);
    _defBytes -= e.bytes;
    _def.delete(k);
  }
}

export function hostReadHook(hostArray) {
  const s = _safe.get(hostArray);
  if (s) s.valid = false;
  const d = _def.get(hostArray);
  if (d) {
    if (d.hostStale) { _syncDownload(hostArray, d.dptr); d.hostStale = false; }
    d.deviceFresh = false;
  }
}

setEagerFlushHook(flushDeferred);
setEagerDeferred(true);
