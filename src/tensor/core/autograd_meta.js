export class AutogradMeta {
  constructor() {
    this._grad = null;
    this._gradFn = null;
    this._outputNr = 0;
    this._requiresGrad = false;
    this._retainGrad = false;
    this._gradAccumulator = null;
    this._versionAtCreation = 0;
  }

  get grad() {
    return this._grad;
  }

  set grad(tensor) {
    this._grad = tensor;
  }

  get gradFn() {
    return this._gradFn;
  }

  setGradFn(node, outputNr) {
    this._gradFn = node;
    this._outputNr = outputNr ?? 0;
  }

  get outputNr() {
    return this._outputNr;
  }

  get requiresGrad() {
    return this._requiresGrad;
  }

  set requiresGrad(flag) {
    this._requiresGrad = flag;
  }

  get retainGrad() {
    return this._retainGrad;
  }

  set retainGrad(flag) {
    this._retainGrad = flag;
  }

  get isLeaf() {
    return this._gradFn === null;
  }

  get versionAtCreation() {
    return this._versionAtCreation;
  }

  set versionAtCreation(v) {
    this._versionAtCreation = v;
  }

  getGradAccumulator() {
    if (this._gradAccumulator) {
      const acc = this._gradAccumulator.deref ? this._gradAccumulator.deref() : this._gradAccumulator;
      if (acc) return acc;
    }
    return null;
  }

  setGradAccumulator(acc) {
    this._gradAccumulator = typeof WeakRef !== 'undefined' ? new WeakRef(acc) : acc;
  }

  clearGrad() {
    this._grad = null;
  }

  clearGradFn() {
    this._gradFn = null;
    this._outputNr = 0;
  }
}
