import { collectBufferAccesses, coversWholeBuffer, AccessKind } from './buffer_access.js';

export const StorageRequirement = Object.freeze({
  FRESH: 'fresh',
  DEFINED: 'defined',
});

export class StorageFact {
  constructor(buffer, requirement, reason) {
    this.buffer = buffer;
    this.requirement = requirement;
    this.reason = reason;
  }

  get needsDefinedStorage() { return this.requirement === StorageRequirement.DEFINED; }
}

function classify(buffer, accesses) {
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

export function analyzeStorageRequirements(primFunc) {
  const { byBuffer } = collectBufferAccesses(primFunc.body);
  const facts = new Map();
  for (const [buffer, accesses] of byBuffer) {
    facts.set(buffer, classify(buffer, accesses));
  }
  return facts;
}

export function buffersRequiringDefinedStorage(primFunc) {
  const required = new Set();
  for (const [buffer, fact] of analyzeStorageRequirements(primFunc)) {
    if (fact.needsDefinedStorage) required.add(buffer);
  }
  return required;
}
