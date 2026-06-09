import { Callback } from './callback.js';

export class EarlyStopping extends Callback {
  constructor({
    monitor = 'val_loss',
    patience = 3,
    mode = 'min',
    minDelta = 0,
    checkOnTrainEpochEnd = false,
  } = {}) {
    super();
    this._monitor = monitor;
    this._patience = patience;
    this._mode = mode;
    this._minDelta = minDelta;
    this._checkOnTrainEpochEnd = checkOnTrainEpochEnd;
    this._waitCount = 0;
    this._bestScore = null;
    this._compareFn = mode === 'min'
      ? (current, best) => current < best - minDelta
      : (current, best) => current > best + minDelta;
  }

  get monitor() { return this._monitor; }
  get patience() { return this._patience; }
  get bestScore() { return this._bestScore; }
  get waitCount() { return this._waitCount; }

  onValidationEnd(trainer, _model) {
    if (this._checkOnTrainEpochEnd) return;
    this._check(trainer);
  }

  onTrainEpochEnd(trainer, _model) {
    if (!this._checkOnTrainEpochEnd) return;
    this._check(trainer);
  }

  _check(trainer) {
    const metrics = trainer.state.epochMetrics.computeAll();
    const current = metrics[this._monitor];
    if (current === undefined) return;

    if (this._bestScore === null || this._compareFn(current, this._bestScore)) {
      this._bestScore = current;
      this._waitCount = 0;
      return;
    }

    this._waitCount++;
    if (this._waitCount >= this._patience) {
      trainer.shouldStop = true;
    }
  }

  reset() {
    this._waitCount = 0;
    this._bestScore = null;
  }
}
