export class Metric {
  constructor() {
    this._computed = null;
  }

  update(_preds, _target) {
    throw new Error(`${this.constructor.name}.update() not implemented`);
  }

  compute() {
    throw new Error(`${this.constructor.name}.compute() not implemented`);
  }

  reset() {
    this._computed = null;
  }

  forward(preds, target) {
    this.update(preds, target);
    this._computed = this.compute();
    return this._computed;
  }

  get value() {
    return this._computed;
  }
}
