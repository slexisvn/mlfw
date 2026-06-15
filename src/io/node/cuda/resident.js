import { acquire, release, copyHostToDevice, copyDeviceToHost } from './memory.js';

const CAPACITY = 1024;
const _cache = new Map();

function _entryFor(hostArray) {
  let e = _cache.get(hostArray);
  if (e) {
    _cache.delete(hostArray);
    _cache.set(hostArray, e);
    return e;
  }
  if (_cache.size >= CAPACITY) {
    const oldestKey = _cache.keys().next().value;
    const oldest = _cache.get(oldestKey);
    _cache.delete(oldestKey);
    release(oldest.dptr, oldest.bytes);
  }
  e = { dptr: acquire(hostArray.byteLength), bytes: hostArray.byteLength, valid: false };
  _cache.set(hostArray, e);
  return e;
}

export function uploadIfStale(hostArray) {
  const e = _entryFor(hostArray);
  if (!e.valid) {
    copyHostToDevice(e.dptr, hostArray);
    e.valid = true;
  }
  return e.dptr;
}

export function downloadAndValidate(hostArray, dptr) {
  copyDeviceToHost(hostArray, dptr);
  const e = _cache.get(hostArray);
  if (e) e.valid = true;
}

export function invalidate(hostArray) {
  const e = _cache.get(hostArray);
  if (e) e.valid = false;
}
