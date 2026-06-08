export class GradAccumulator {
  constructor(builder) {
    this._builder = builder;
    this._grads = new Map();
  }

  accumulate(valueId, gradValue) {
    if (!gradValue) return;
    const existing = this._grads.get(valueId);
    if (!existing) {
      this._grads.set(valueId, gradValue);
    } else {
      const summed = this._builder.add(existing, gradValue).getResult(0);
      this._grads.set(valueId, summed);
    }
  }

  get(valueId) {
    return this._grads.get(valueId) || null;
  }

  has(valueId) {
    return this._grads.has(valueId);
  }
}
