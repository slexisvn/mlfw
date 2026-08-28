import { fuseLocations } from '../location.js';
import type { Location } from '../location.js';
import type { Operation } from './operation.js';
import type { Value } from './value.js';

export function opsLocation(ops: Iterable<Operation>, tag: string | null = null): Location | null {
  const locations: (Location | null)[] = [];
  for (const op of ops) locations.push(op.loc);
  return fuseLocations(locations, tag);
}

export function producerLocation(values: Iterable<Value>, tag: string | null = null): Location | null {
  const locations: (Location | null)[] = [];
  for (const value of values) locations.push(value.definingOp ? value.definingOp.loc : null);
  return fuseLocations(locations, tag);
}

export function derivedFrom<T extends Operation>(op: T, source: Operation | null): T {
  op.loc = source === null ? null : source.loc;
  return op;
}
