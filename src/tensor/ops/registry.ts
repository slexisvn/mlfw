import { dispatcher } from '../../dispatcher/dispatcher.js';
import type { OperatorHandle } from '../../dispatcher/operator_handle.js';

const _cache = new Map<string, OperatorHandle>();

export function getHandle(name: string): OperatorHandle | null {
  let h: OperatorHandle | null = _cache.get(name) ?? null;
  if (!h) {
    h = dispatcher.findOp(name);
    if (h) _cache.set(name, h);
  }
  return h || null;
}
