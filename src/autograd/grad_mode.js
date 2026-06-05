let _enabled = true;

export const GradMode = {
  isEnabled() { return _enabled; },
  setEnabled(flag) { _enabled = flag; },
};

export function noGrad(fn) {
  const prev = _enabled;
  _enabled = false;
  try {
    return fn();
  } finally {
    _enabled = prev;
  }
}

export function enableGrad(fn) {
  const prev = _enabled;
  _enabled = true;
  try {
    return fn();
  } finally {
    _enabled = prev;
  }
}
