import { acquire, release, copyHostToDevice, copyDeviceToHost, copyHostToDeviceAsync, copyDeviceToHostAsync } from './memory.js';
import { getDevice } from './device.js';
import { cu, checkCU } from './ffi.js';
import { setEagerDeferred, isEagerDeferred, setEagerFlushHook, isEagerCapturing, setCudaGraphArmed } from '../../../dispatcher/eager_mode.js';

export { setEagerDeferred, isEagerDeferred, setCudaGraphArmed };

const CACHE_FRACTION_OF_FREE = 0.15;
const MIN_CACHE_BYTES = 128 * 1024 * 1024;
const LOW_WATER = 0.75;

const _entries = new Map();
const _byHost = new WeakMap();
let _pinned = new WeakSet();
let _capturePinned = new WeakSet();
let _defBytes = 0;
let _defCap = 0;
let _nextId = 0;

const _reclaim = new FinalizationRegistry((entry) => _reclaimEntry(entry));

export function pinResident(hostArray) { _pinned.add(hostArray); }
export function unpinResident(hostArray) { _pinned.delete(hostArray); }
export function clearCapturePins() { _capturePinned = new WeakSet(); }
export function deviceBufferDptr(hostArray) { const e = _byHost.get(hostArray); return e ? e.dptr : null; }
export function uploadStaticAsync(dptr, hostArray) { copyHostToDeviceAsync(dptr, hostArray); }
export function downloadStaticAsync(hostArray, dptr) { copyDeviceToHostAsync(hostArray, dptr); }
export function residentBytes() { return _defBytes; }

export function setResidentCacheLimit(bytes) { _defCap = Math.max(bytes, 0); }

function _cap() {
  if (!_defCap) {
    getDevice();
    const free = [0n], total = [0n];
    checkCU('cuMemGetInfo', cu.memGetInfo(free, total));
    _defCap = Math.max(Math.floor(Number(free[0]) * CACHE_FRACTION_OF_FREE), MIN_CACHE_BYTES);
  }
  return _defCap;
}

function _syncDownload(hostArray, dptr) {
  if (isEagerCapturing()) throw new Error('illegal device sync during CUDA graph capture');
  checkCU('cuStreamSynchronize', cu.streamSynchronize(getDevice().stream));
  copyDeviceToHost(hostArray, dptr);
}

function _reclaimEntry(entry) {
  if (entry.freed) return;
  entry.freed = true;
  _entries.delete(entry.id);
  _defBytes -= entry.bytes;
  release(entry.dptr, entry.bytes);
}

function _freeEntry(entry, writeBack) {
  if (entry.freed) return;
  const host = entry.ref.deref();
  if (host !== undefined) {
    if (writeBack && entry.hostStale) _syncDownload(host, entry.dptr);
    _reclaim.unregister(host);
    _byHost.delete(host);
  }
  _reclaimEntry(entry);
}

function _evictPass(target, incoming, evictDirty) {
  for (const entry of _entries.values()) {
    if (_defBytes + incoming <= target) return true;
    const host = entry.ref.deref();
    if (host === undefined) { _freeEntry(entry, false); continue; }
    if (_pinned.has(host) || _capturePinned.has(host)) continue;
    if (entry.hostStale && !evictDirty) continue;
    _freeEntry(entry, evictDirty);
  }
  return _defBytes + incoming <= target;
}

function _evict(incoming) {
  const cap = _cap();
  if (_defBytes + incoming <= cap) return;
  const target = Math.floor(cap * LOW_WATER);
  if (_evictPass(target, incoming, false)) return;
  _evictPass(target, incoming, true);
}

function _defEntry(hostArray) {
  let e = _byHost.get(hostArray);
  if (e) {
    _entries.delete(e.id);
    _entries.set(e.id, e);
  } else {
    if (!isEagerCapturing()) _evict(hostArray.byteLength);
    e = {
      id: _nextId++,
      ref: new WeakRef(hostArray),
      dptr: acquire(hostArray.byteLength),
      bytes: hostArray.byteLength,
      deviceFresh: false,
      hostStale: false,
      freed: false,
    };
    _byHost.set(hostArray, e);
    _entries.set(e.id, e);
    _defBytes += e.bytes;
    _reclaim.register(hostArray, e, hostArray);
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
  for (const entry of _entries.values()) {
    const host = entry.ref.deref();
    if (host !== undefined && (_pinned.has(host) || _capturePinned.has(host))) continue;
    _freeEntry(entry, false);
  }
}

export function releaseAllResident() {
  for (const entry of _entries.values()) _freeEntry(entry, true);
  _entries.clear();
  _defBytes = 0;
  _pinned = new WeakSet();
  _capturePinned = new WeakSet();
}

export function hostReadHook(hostArray) {
  const d = _byHost.get(hostArray);
  if (isEagerCapturing()) {
    if (d && d.hostStale) throw new Error('illegal host read of device-resident tensor during CUDA graph capture');
    return;
  }
  if (!d) return;
  if (d.hostStale) { _syncDownload(hostArray, d.dptr); d.hostStale = false; }
  d.deviceFresh = false;
}

setEagerFlushHook(flushDeferred);
setEagerDeferred(true);
