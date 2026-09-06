import { AxeAxis } from '../ir/layout/axe.js';
import type { AxeAxisName } from '../ir/layout/axe.js';
import type { LIRThreadBinding } from '../ir/lir/nodes.js';

export type ThreadSpace = 'thread' | 'block';
export type ThreadAxis = { space: ThreadSpace; axis: number };

const AXIS_BASE = 120;

const AXE_AXIS_BY_SPACE: Readonly<Record<ThreadSpace, readonly AxeAxisName[]>> = Object.freeze({
  thread: [AxeAxis.THREAD_X, AxeAxis.THREAD_Y, AxeAxis.THREAD_Z],
  block: [AxeAxis.BLOCK_X, AxeAxis.BLOCK_Y, AxeAxis.BLOCK_Z]
});

const TAG_BY_AXE_AXIS: ReadonlyMap<AxeAxisName, string> = new Map(
  (['thread', 'block'] as const).flatMap(space =>
    AXE_AXIS_BY_SPACE[space].map((axis, i) => [axis, `${space === 'thread' ? 'threadIdx' : 'blockIdx'}.${'xyz'[i]}`] as const))
);

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

export function axeAxisOfThreadTag(tag: string): AxeAxisName | null {
  const parsed = parseThreadAxis(tag);
  return parsed ? AXE_AXIS_BY_SPACE[parsed.space][parsed.axis] : null;
}

export function threadTagOfAxeAxis(axis: AxeAxisName): string | null {
  return TAG_BY_AXE_AXIS.get(axis) ?? null;
}

export function maxBindingExtent(threadBindings: ReadonlyMap<string, readonly LIRThreadBinding[]>, tag: string | null): number {
  const entries = tag === null ? undefined : threadBindings.get(tag);
  if (!entries) return 0;
  let max = 0;
  for (const e of entries) if (e.extent > max) max = e.extent;
  return max;
}
