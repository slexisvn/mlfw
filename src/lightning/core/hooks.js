export const HOOKS = Object.freeze({
  ON_FIT_START: 'onFitStart',
  ON_FIT_END: 'onFitEnd',
  ON_TRAIN_START: 'onTrainStart',
  ON_TRAIN_END: 'onTrainEnd',
  ON_TRAIN_EPOCH_START: 'onTrainEpochStart',
  ON_TRAIN_EPOCH_END: 'onTrainEpochEnd',
  ON_TRAIN_BATCH_START: 'onTrainBatchStart',
  ON_TRAIN_BATCH_END: 'onTrainBatchEnd',
  ON_VALIDATION_START: 'onValidationStart',
  ON_VALIDATION_END: 'onValidationEnd',
  ON_VALIDATION_EPOCH_START: 'onValidationEpochStart',
  ON_VALIDATION_EPOCH_END: 'onValidationEpochEnd',
  ON_VALIDATION_BATCH_START: 'onValidationBatchStart',
  ON_VALIDATION_BATCH_END: 'onValidationBatchEnd',
  ON_TEST_START: 'onTestStart',
  ON_TEST_END: 'onTestEnd',
  ON_TEST_BATCH_START: 'onTestBatchStart',
  ON_TEST_BATCH_END: 'onTestBatchEnd',
  ON_PREDICT_START: 'onPredictStart',
  ON_PREDICT_END: 'onPredictEnd',
  ON_PREDICT_BATCH_START: 'onPredictBatchStart',
  ON_PREDICT_BATCH_END: 'onPredictBatchEnd',
  SETUP: 'setup',
  TEARDOWN: 'teardown',
  ON_BEFORE_BACKWARD: 'onBeforeBackward',
  ON_AFTER_BACKWARD: 'onAfterBackward',
  ON_BEFORE_OPTIMIZER_STEP: 'onBeforeOptimizerStep',
  ON_BEFORE_ZERO_GRAD: 'onBeforeZeroGrad',
  ON_SAVE_CHECKPOINT: 'onSaveCheckpoint',
  ON_LOAD_CHECKPOINT: 'onLoadCheckpoint',
});

export class CallbackConnector {
  constructor(callbacks = []) {
    this._callbacks = callbacks;
  }

  get callbacks() {
    return this._callbacks;
  }

  add(callback) {
    this._callbacks.push(callback);
  }

  dispatch(hookName, ...args) {
    for (let i = 0; i < this._callbacks.length; i++) {
      const fn = this._callbacks[i][hookName];
      if (fn) fn.call(this._callbacks[i], ...args);
    }
  }

  remove(callback) {
    const idx = this._callbacks.indexOf(callback);
    if (idx !== -1) this._callbacks.splice(idx, 1);
  }
}

export class LoggerConnector {
  constructor(loggers = [], state = null) {
    this._loggers = Array.isArray(loggers) ? loggers : [loggers];
    this._state = state;
  }

  drain(model) {
    const buf = model._logBuffer;
    if (buf.size === 0) return;

    for (const [name, entry] of buf) {
      const { value, onStep, onEpoch, reduceFx, progBar } = entry;

      if (onEpoch) {
        this._state.epochMetrics.update(name, value, reduceFx);
      }

      if (onStep) {
        this._state.stepMetrics.update(name, value, reduceFx);
      }

      if (progBar) {
        if (!this._state._progBarMetrics) this._state._progBarMetrics = new Map();
        const v = typeof value === 'number' ? value : value.item();
        this._state._progBarMetrics.set(name, v);
      }
    }

    buf.clear();
  }

  flushStepMetrics(step) {
    const metrics = this._state.stepMetrics.computeAll();
    if (Object.keys(metrics).length === 0) return metrics;
    for (let i = 0; i < this._loggers.length; i++) {
      this._loggers[i].logMetrics(metrics, step);
    }
    this._state.stepMetrics.reset();
    return metrics;
  }

  flushEpochMetrics(step) {
    const metrics = this._state.epochMetrics.computeAll();
    if (Object.keys(metrics).length === 0) return metrics;
    for (let i = 0; i < this._loggers.length; i++) {
      this._loggers[i].logMetrics(metrics, step);
    }
    this._state.epochMetrics.reset();
    return metrics;
  }

  logHyperparams(params) {
    for (let i = 0; i < this._loggers.length; i++) {
      this._loggers[i].logHyperparams(params);
    }
  }
}
