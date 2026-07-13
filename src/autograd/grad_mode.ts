let _enabled = true;

export const GradMode = {
  isEnabled() { return _enabled; },
  setEnabled(flag: boolean) { _enabled = flag; },
};

export function noGrad<T>(fn: () => T): T {
  const prev = _enabled;
  _enabled = false;
  try {
    return fn();
  } finally {
    _enabled = prev;
  }
}

export function enableGrad<T>(fn: () => T): T {
  const prev = _enabled;
  _enabled = true;
  try {
    return fn();
  } finally {
    _enabled = prev;
  }
}
