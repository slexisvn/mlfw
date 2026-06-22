let _deferred = false;
let _flushHook = null;
let _capturing = false;
let _graphArmed = false;

export function setEagerDeferred(on) { _deferred = on; }
export function isEagerDeferred() { return _deferred; }

export function setEagerCapturing(on) { _capturing = on; }
export function isEagerCapturing() { return _capturing; }

export function setCudaGraphArmed(on) { _graphArmed = on; }
export function isCudaGraphArmed() { return _graphArmed; }

export function setEagerFlushHook(fn) { _flushHook = fn; }
export function eagerFlush() { if (_flushHook) _flushHook(); }
