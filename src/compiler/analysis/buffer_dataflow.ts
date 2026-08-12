import { collectBufferAccesses, coversWholeBuffer, AccessKind } from './buffer_access.js';

export const StorageRequirement = Object.freeze({
  FRESH: 'fresh',
  DEFINED: 'defined',
});

import type { BufferAccess } from './buffer_access.js';
import type { Buffer } from '../ir/tensor/buffer.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';

export type StorageRequirementValue = (typeof StorageRequirement)[keyof typeof StorageRequirement];

export class StorageFact {
  buffer: Buffer;
  requirement: StorageRequirementValue;
  reason: string;

  constructor(buffer: Buffer, requirement: StorageRequirementValue, reason: string) {
    this.buffer = buffer;
    this.requirement = requirement;
    this.reason = reason;
  }

  get needsDefinedStorage(): boolean { return this.requirement === StorageRequirement.DEFINED; }
}

function classify(buffer: Buffer, accesses: readonly BufferAccess[]): StorageFact {
  for (const access of accesses) {
    if (access.selfReferential) {
      return new StorageFact(buffer, StorageRequirement.DEFINED,
        'read-modify-write: a store reads the buffer it writes');
    }
  }

  const firstRead = accesses.findIndex((a) => a.kind === AccessKind.READ);
  if (firstRead === -1) {
    return new StorageFact(buffer, StorageRequirement.FRESH, 'never read');
  }

  const firstWrite = accesses.findIndex((a) => a.kind === AccessKind.WRITE);
  if (firstWrite === -1 || firstWrite > firstRead) {
    return new StorageFact(buffer, StorageRequirement.DEFINED,
      'upward-exposed use: read before any write');
  }

  for (let i = 0; i < firstRead; i++) {
    const access = accesses[i];
    if (access.kind !== AccessKind.WRITE || access.conditional) continue;
    if (coversWholeBuffer(access)) {
      return new StorageFact(buffer, StorageRequirement.FRESH,
        'fully defined by an unconditional write before its first read');
    }
  }

  return new StorageFact(buffer, StorageRequirement.DEFINED,
    'no unconditional write proven to cover the buffer before its first read');
}

export function analyzeStorageRequirements(primFunc: PrimFunc): Map<Buffer, StorageFact> {
  const { byBuffer } = collectBufferAccesses(primFunc.body);
  const facts = new Map<Buffer, StorageFact>();
  for (const [buffer, accesses] of byBuffer) {
    facts.set(buffer, classify(buffer, accesses));
  }
  return facts;
}

export function buffersRequiringDefinedStorage(primFunc: PrimFunc): Set<Buffer> {
  const required = new Set<Buffer>();
  for (const [buffer, fact] of analyzeStorageRequirements(primFunc)) {
    if (fact.needsDefinedStorage) required.add(buffer);
  }
  return required;
}
