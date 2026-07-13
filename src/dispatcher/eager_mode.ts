let _deferred = false;
let _flushHook: (() => void) | null = null;
let _capturing = false;
let _graphArmed = false;

export function setEagerDeferred(on: boolean): void { _deferred = on; }
export function isEagerDeferred(): boolean { return _deferred; }

export function setEagerCapturing(on: boolean): void { _capturing = on; }
export function isEagerCapturing(): boolean { return _capturing; }

export function setCudaGraphArmed(on: boolean): void { _graphArmed = on; }
export function isCudaGraphArmed(): boolean { return _graphArmed; }

export function setEagerFlushHook(fn: (() => void) | null): void { _flushHook = fn; }
export function eagerFlush(): void { if (_flushHook) _flushHook(); }
