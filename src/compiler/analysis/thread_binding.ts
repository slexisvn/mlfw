import type { LIRThreadBinding } from '../ir/lir/nodes.js';

export type ThreadSpace = 'thread' | 'block';
export type ThreadAxis = { space: ThreadSpace; axis: number };

const AXIS_BASE = 120;

export function parseThreadAxis(tag: string): ThreadAxis | null {
  const idx = tag.indexOf('.');
  if (idx < 0) return null;
  const axis = tag.charCodeAt(idx + 1) - AXIS_BASE;
  if (axis < 0 || axis > 2) return null;
  const prefix = tag.substring(0, idx);
  if (prefix === 'threadIdx') return { space: 'thread', axis };
  if (prefix === 'blockIdx') return { space: 'block', axis };
  return null;
}

export function maxBindingExtent(threadBindings: ReadonlyMap<string, readonly LIRThreadBinding[]>, tag: string | null): number {
  const entries = tag === null ? undefined : threadBindings.get(tag);
  if (!entries) return 0;
  let max = 0;
  for (const e of entries) if (e.extent > max) max = e.extent;
  return max;
}
