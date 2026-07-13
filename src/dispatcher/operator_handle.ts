import type { DispatchKeySet, DispatchKeyValue } from './dispatch_key.js';
import type { KernelFunction } from './boxing.js';
import type { OperatorEntry, KernelLookup } from './operator_entry.js';
import type { OperatorSchema } from './operator_schema.js';

export class OperatorHandle {
  private readonly _entry: OperatorEntry;
  private readonly _schema: OperatorSchema;

  constructor(entry: OperatorEntry, schema: OperatorSchema) {
    this._entry = entry;
    this._schema = schema;
  }

  get entry(): OperatorEntry {
    return this._entry;
  }

  get schema(): OperatorSchema {
    return this._schema;
  }

  get name(): string {
    return this._schema.name;
  }

  get qualifiedName(): string {
    return this._schema.qualifiedName();
  }

  get key(): string {
    return this._schema.key();
  }

  get tensorArgIndices(): readonly number[] {
    return this._schema.tensorArgIndices;
  }

  lookupKernel(dispatchKey: DispatchKeyValue): KernelFunction | null {
    return this._entry.lookupKernel(dispatchKey);
  }

  bestKernel(keySet: DispatchKeySet): KernelLookup | null {
    return this._entry.bestKernel(keySet);
  }
}
