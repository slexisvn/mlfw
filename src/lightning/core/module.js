import { Module } from '../../nn/module.js';

export class LightningModule extends Module {
  constructor() {
    super();
    this._trainer = null;
    this._logBuffer = new Map();
    this._automaticOptimization = true;
    this._currentOptimizers = [];
    this._device = null;
  }

  get trainer() {
    return this._trainer;
  }

  get currentEpoch() {
    return this._trainer ? this._trainer.state.epoch : 0;
  }

  get globalStep() {
    return this._trainer ? this._trainer.state.globalStep : 0;
  }

  get device() {
    return this._device;
  }

  get logger() {
    return this._trainer ? this._trainer.logger : null;
  }

  get loggers() {
    return this._trainer ? this._trainer.loggers : [];
  }

  get automaticOptimization() {
    return this._automaticOptimization;
  }

  set automaticOptimization(value) {
    this._automaticOptimization = value;
  }

  get optimizers() {
    return this._currentOptimizers;
  }

  trainingStep(_batch, _batchIdx) {
    throw new Error(`${this.constructor.name}.trainingStep() not implemented`);
  }

  validationStep(_batch, _batchIdx) {}

  testStep(_batch, _batchIdx) {}

  predictStep(batch, _batchIdx) {
    return this.forward(Array.isArray(batch) && batch.length === 1 ? batch[0] : batch);
  }

  configureOptimizers() {
    throw new Error(`${this.constructor.name}.configureOptimizers() not implemented`);
  }

  onTrainEpochStart() {}
  onTrainEpochEnd() {}
  onValidationEpochStart() {}
  onValidationEpochEnd() {}
  onTestEpochStart() {}
  onTestEpochEnd() {}

  log(name, value, {
    onStep = null,
    onEpoch = null,
    reduceFx = 'mean',
    progBar = false,
  } = {}) {
    const stage = this._trainer ? this._trainer.state.stage : 'training';
    const stepDefault = stage === 'training';
    const epochDefault = stage !== 'training';
    this._logBuffer.set(name, {
      value,
      onStep: onStep !== null ? onStep : stepDefault,
      onEpoch: onEpoch !== null ? onEpoch : epochDefault,
      reduceFx,
      progBar,
    });
  }

  logDict(dict, opts = {}) {
    for (const key of Object.keys(dict)) {
      this.log(key, dict[key], opts);
    }
  }

  manualBackward(loss) {
    if (this._trainer && this._trainer.strategy) {
      this._trainer.strategy.backward(loss);
    } else {
      loss.backward();
    }
  }
}

export function parseOptimizersConfig(result) {
  if (!result) throw new Error('configureOptimizers() returned null/undefined');

  if (Array.isArray(result)) {
    const optimizers = [];
    const schedulerConfigs = [];
    for (let i = 0; i < result.length; i++) {
      const item = result[i];
      if (item.optimizer) {
        optimizers.push(item.optimizer);
        schedulerConfigs.push(normalizeSchedulerConfig(item.lrScheduler));
      } else {
        optimizers.push(item);
        schedulerConfigs.push(null);
      }
    }
    return { optimizers, schedulerConfigs };
  }

  if (result.optimizer) {
    return {
      optimizers: [result.optimizer],
      schedulerConfigs: [normalizeSchedulerConfig(result.lrScheduler)],
    };
  }

  if (typeof result.step === 'function') {
    return { optimizers: [result], schedulerConfigs: [null] };
  }

  throw new Error('configureOptimizers() returned an unrecognized format');
}

function normalizeSchedulerConfig(config) {
  if (!config) return null;
  if (config.scheduler) {
    return {
      scheduler: config.scheduler,
      interval: config.interval || 'epoch',
      frequency: config.frequency || 1,
      monitor: config.monitor || null,
    };
  }
  if (typeof config.step === 'function') {
    return {
      scheduler: config,
      interval: 'epoch',
      frequency: 1,
      monitor: null,
    };
  }
  return null;
}
