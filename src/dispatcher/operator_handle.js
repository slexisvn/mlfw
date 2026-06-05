export class OperatorHandle {
  constructor(entry, schema) {
    this._entry = entry;
    this._schema = schema;
  }

  get entry() {
    return this._entry;
  }

  get schema() {
    return this._schema;
  }

  get name() {
    return this._schema.name;
  }

  get qualifiedName() {
    return this._schema.qualifiedName();
  }

  get key() {
    return this._schema.key();
  }

  get tensorArgIndices() {
    return this._schema.tensorArgIndices;
  }

  lookupKernel(dispatchKey) {
    return this._entry.lookupKernel(dispatchKey);
  }

  bestKernel(keySet) {
    return this._entry.bestKernel(keySet);
  }
}
