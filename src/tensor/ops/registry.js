import { dispatcher } from '../../dispatcher/dispatcher.js';

const _cache = new Map();

export function getHandle(name) {
  let h = _cache.get(name);
  if (!h) {
    h = dispatcher.findOp(name);
    if (h) _cache.set(name, h);
  }
  return h;
}

export function clearHandleCache() {
  _cache.clear();
}
