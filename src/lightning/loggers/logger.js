export class Logger {
  constructor({ name = 'default', version = 0 } = {}) {
    this._name = name;
    this._version = version;
  }

  get name() {
    return this._name;
  }

  get version() {
    return this._version;
  }

  logMetrics(_metrics, _step) {
    throw new Error(`${this.constructor.name}.logMetrics() not implemented`);
  }

  logHyperparams(_params) {
    throw new Error(`${this.constructor.name}.logHyperparams() not implemented`);
  }

  finalize() {}
}
