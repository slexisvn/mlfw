let _deferred = false;
let _flushHook = null;

export function setEagerDeferred(on) { _deferred = on; }
export function isEagerDeferred() { return _deferred; }

export function setEagerFlushHook(fn) { _flushHook = fn; }
export function eagerFlush() { if (_flushHook) _flushHook(); }
