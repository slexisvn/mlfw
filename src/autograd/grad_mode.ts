import { scoped } from '../util/scoped.js';

let _enabled = true;

export const GradMode = {
  isEnabled() { return _enabled; },
  setEnabled(flag: boolean) { _enabled = flag; },
};

function withGradEnabled<T>(flag: boolean, fn: () => T): T {
  const prev = _enabled;
  _enabled = flag;
  return scoped(fn, () => { _enabled = prev; });
}

export function noGrad<T>(fn: () => T): T {
  return withGradEnabled(false, fn);
}

export function enableGrad<T>(fn: () => T): T {
  return withGradEnabled(true, fn);
}
