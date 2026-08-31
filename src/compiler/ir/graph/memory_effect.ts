import type { SideEffectMask } from './op_registry.js';
import type { Value } from './value.js';

export class MemoryEffect {
  kind: SideEffectMask;
  value: Value;

  constructor(kind: SideEffectMask, value: Value) {
    this.kind = kind;
    this.value = value;
  }
}
