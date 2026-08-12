import { registry } from './ops.js';
import type { Operation } from './operation.js';

export const OpAttrKey = Object.freeze({
  GPU_CAPABLE: 'gpuCapable',
  LAUNCH_BOUNDARY: 'launchBoundary',
  INFER_LAYOUT: 'inferLayout',
  LAYOUT_SENSITIVITY: 'layoutSensitivity',
});

export function launchBoundaryClass(opName: string): string | null {
  const def = registry.get(opName);
  return def === null ? null : def.getAttr<string>(OpAttrKey.LAUNCH_BOUNDARY);
}

export function isLaunchBoundaryOp(opName: string): boolean {
  return launchBoundaryClass(opName) !== null;
}

export function containsLaunchBoundary(op: Operation): boolean {
  if (isLaunchBoundaryOp(op.opName)) return true;
  if (!op.regions) return false;
  for (const region of op.regions) {
    const block = region.entryBlock;
    if (!block) continue;
    for (const inner of block.ops()) {
      if (containsLaunchBoundary(inner)) return true;
    }
  }
  return false;
}

export function countLaunchBoundaries(ops: Iterable<Operation>): Map<string, number> {
  const counts = new Map<string, number>();
  for (const op of ops) {
    const cls = launchBoundaryClass(op.opName);
    if (cls === null) continue;
    counts.set(cls, (counts.get(cls) || 0) + 1);
  }
  return counts;
}

export function isConstantOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isConstant;
}

export function isTerminatorOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isTerminator;
}

export function isBroadcastOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isBroadcast;
}

export function isReductionOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isReduction;
}
