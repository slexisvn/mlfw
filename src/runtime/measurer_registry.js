const _measurers = new Map();

export function registerMeasurer(kind, fn) {
  _measurers.set(kind, fn);
}

export function getMeasurer(kind) {
  return _measurers.get(kind) || null;
}

export function hasMeasurer(kind) {
  return _measurers.has(kind);
}
