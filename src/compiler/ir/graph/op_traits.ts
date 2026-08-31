import { registry } from './ops.js';
import { OpAttrKey, OpTrait } from './op_registry.js';
import type { Operation } from './operation.js';

export { OpAttrKey } from './op_registry.js';

export function unifiedOperandIndices(opName: string, numOperands: number): readonly number[] | null {
  const def = registry.get(opName);
  if (def === null) return null;
  const declared = def.getAttr<readonly number[]>(OpAttrKey.UNIFIED_OPERANDS);
  if (declared !== null) return declared;
  if (!def.hasTrait(OpTrait.SAME_OPERAND_AND_RESULT_TYPE)) return null;
  return Array.from({ length: numOperands }, (_, i) => i);
}

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

export function containsSequentialRegion(op: Operation): boolean {
  const def = registry.get(op.opName);
  if (def !== null && def.getAttr<boolean>(OpAttrKey.SEQUENTIAL_REGION) === true) return true;
  if (!op.regions) return false;
  for (const region of op.regions) {
    const block = region.entryBlock;
    if (!block) continue;
    for (const inner of block.ops()) {
      if (containsSequentialRegion(inner)) return true;
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

export function opHasSideEffects(op: Operation): boolean {
  const def = registry.get(op.opName);
  if (def !== null && def.hasSideEffects) return true;
  if (def === null || !def.hasRecursiveMemoryEffects || !op.regions) return false;
  for (const region of op.regions) {
    const block = region.entryBlock;
    if (!block) continue;
    for (const inner of block.ops()) {
      if (opHasSideEffects(inner)) return true;
    }
  }
  return false;
}

export function effectPredecessors(ops: Iterable<Operation>): Map<Operation, Operation> {
  const chain = new Map<Operation, Operation>();
  let previous: Operation | null = null;
  for (const op of ops) {
    if (!opHasSideEffects(op)) continue;
    if (previous !== null) chain.set(op, previous);
    previous = op;
  }
  return chain;
}

export function isConstantOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isConstant;
}

export function isTerminatorOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isTerminator;
}

export function isElementwiseOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isElementwise;
}

export function isBroadcastOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isBroadcast;
}

export function isReductionOp(opName: string): boolean {
  const def = registry.get(opName);
  return def !== null && def.isReduction;
}

export type LibraryTarget = { hasLibraryClass(cls: string | null): boolean };

export function hasLibraryOp(target: LibraryTarget, opName: string): boolean {
  return target.hasLibraryClass(launchBoundaryClass(opName));
}
