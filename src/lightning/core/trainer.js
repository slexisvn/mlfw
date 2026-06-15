import { TrainerState, SingleDeviceStrategy, Stage } from './state.js';
import { CallbackConnector, LoggerConnector } from './hooks.js';
import { FitLoop } from './loops/fit_loop.js';
import { EvaluationLoop } from './loops/evaluation_loop.js';
import { PredictionLoop } from './loops/prediction_loop.js';
import { parseOptimizersConfig } from './module.js';
import { ConsoleLogger } from '../loggers/console.js';
import { ProgressCallback } from '../callbacks/progress.js';
import { ModelCheckpoint } from '../callbacks/checkpoint.js';
import { CPU_DEVICE, GPU_DEVICE, WASM_DEVICE } from '../../tensor/types/device.js';

export class Trainer {
  constructor({
    maxEpochs = 10,
    maxSteps = -1,
    accelerator = 'auto',
    precision = 'f32',
    callbacks = [],
    logger = true,
    enableCheckpointing = false,
    enableProgress = true,
    gradientClipVal = null,
    gradientClipAlgorithm = 'norm',
    accumulateGradBatches = 1,
    limitTrainBatches = null,
    limitValBatches = null,
    limitTestBatches = null,
    valCheckInterval = 1.0,
    checkValEveryNEpoch = 1,
    logEveryNSteps = 50,
    deterministic = false,
    fastDevRun = false,
    defaultRootDir = './lightning_logs',
    compile = false,
    compileMode = 'separate',
  } = {}) {
    this._state = new TrainerState();
    this._state.maxEpochs = maxEpochs;
    this._state.maxSteps = maxSteps;

    this._compile = compile;
    this._compileMode = compileMode;
    this._accelerator = accelerator;
    this._precision = precision;
    this._gradientClipVal = gradientClipVal;
    this._gradientClipAlgorithm = gradientClipAlgorithm;
    this._accumulateGradBatches = accumulateGradBatches;
    this._limitTrainBatches = limitTrainBatches;
    this._limitValBatches = limitValBatches;
    this._limitTestBatches = limitTestBatches;
    this._valCheckInterval = valCheckInterval;
    this._checkValEveryNEpoch = checkValEveryNEpoch;
    this._logEveryNSteps = logEveryNSteps;
    this._deterministic = deterministic;
    this._defaultRootDir = defaultRootDir;

    if (fastDevRun !== false) {
      const batches = typeof fastDevRun === 'number' ? fastDevRun : 1;
      this._limitTrainBatches = batches;
      this._limitValBatches = batches;
      this._limitTestBatches = batches;
      this._state.maxEpochs = 1;
    }

    this._loggers = this._resolveLoggers(logger);
    this._strategy = new SingleDeviceStrategy();
    this._fitLoop = new FitLoop();
    this._evaluationLoop = new EvaluationLoop();
    this._predictionLoop = new PredictionLoop();

    const userCallbacks = [...callbacks];
    if (enableProgress && !userCallbacks.some(c => c instanceof ProgressCallback)) {
      userCallbacks.push(new ProgressCallback());
    }
    if (enableCheckpointing && !userCallbacks.some(c => c instanceof ModelCheckpoint)) {
      userCallbacks.push(new ModelCheckpoint({ dirpath: defaultRootDir }));
    }
    this._callbackConnector = new CallbackConnector(userCallbacks);
    this._loggerConnector = new LoggerConnector(this._loggers, this._state);
    this._model = null;
  }

  get state() { return this._state; }
  get strategy() { return this._strategy; }
  get callbackConnector() { return this._callbackConnector; }
  get loggerConnector() { return this._loggerConnector; }
  get fitLoop() { return this._fitLoop; }
  get gradientClipVal() { return this._gradientClipVal; }
  get gradientClipAlgorithm() { return this._gradientClipAlgorithm; }
  get compile() { return this._compile; }
  get compileMode() { return this._compileMode; }
  get accumulateGradBatches() { return this._accumulateGradBatches; }
  set accumulateGradBatches(v) { this._accumulateGradBatches = v; }
  get limitTrainBatches() { return this._limitTrainBatches; }
  get limitValBatches() { return this._limitValBatches; }
  get limitTestBatches() { return this._limitTestBatches; }
  get checkValEveryNEpoch() { return this._checkValEveryNEpoch; }
  get logEveryNSteps() { return this._logEveryNSteps; }
  get shouldStop() { return this._state.shouldStop; }
  set shouldStop(v) { this._state.shouldStop = v; }
  get currentEpoch() { return this._state.epoch; }
  get globalStep() { return this._state.globalStep; }
  get logger() { return this._loggers[0] || null; }
  get loggers() { return this._loggers; }
  get callbacks() { return this._callbackConnector.callbacks; }
  get model() { return this._model; }
  get defaultRootDir() { return this._defaultRootDir; }

  async fit(model, trainLoader, valLoader = null) {
    this._model = model;
    model._trainer = this;
    const device = this._resolveDevice();
    model._device = device;
    await this._prepareDevice(device);
    this._strategy.setup(model, device);

    const { optimizers, schedulerConfigs } = parseOptimizersConfig(
      await Promise.resolve(model.configureOptimizers())
    );
    model._currentOptimizers = optimizers;

    this._loggerConnector.logHyperparams(this._extractHyperparams(model, optimizers));

    this._callbackConnector.dispatch('setup', this, model, Stage.TRAINING);
    this._callbackConnector.dispatch('onFitStart', this, model);

    this._state.shouldStop = false;
    await this._fitLoop.run(model, trainLoader, valLoader, this, optimizers, schedulerConfigs);

    this._callbackConnector.dispatch('onFitEnd', this, model);
    this._callbackConnector.dispatch('teardown', this, model, Stage.TRAINING);

    for (let i = 0; i < this._loggers.length; i++) {
      this._loggers[i].finalize();
    }
  }

  async validate(model, dataLoader) {
    model._trainer = this;
    this._model = model;
    const device = this._resolveDevice();
    model._device = device;
    this._strategy.setup(model, device);
    this._callbackConnector.dispatch('setup', this, model, Stage.VALIDATING);
    const metrics = await this._fitLoop.validationLoop.run(model, dataLoader, this, null);
    this._callbackConnector.dispatch('teardown', this, model, Stage.VALIDATING);
    for (let i = 0; i < this._loggers.length; i++) this._loggers[i].finalize();
    return metrics;
  }

  async test(model, dataLoader) {
    model._trainer = this;
    this._model = model;
    const device = this._resolveDevice();
    model._device = device;
    this._strategy.setup(model, device);
    this._callbackConnector.dispatch('setup', this, model, Stage.TESTING);
    const metrics = await this._evaluationLoop.run(model, dataLoader, this);
    this._callbackConnector.dispatch('teardown', this, model, Stage.TESTING);
    for (let i = 0; i < this._loggers.length; i++) this._loggers[i].finalize();
    return metrics;
  }

  async predict(model, dataLoader) {
    model._trainer = this;
    this._model = model;
    const device = this._resolveDevice();
    model._device = device;
    this._strategy.setup(model, device);
    return await this._predictionLoop.run(model, dataLoader, this);
  }

  _resolveDevice() {
    if (this._accelerator === 'gpu') return GPU_DEVICE;
    if (this._accelerator === 'wasm') return WASM_DEVICE;
    if (this._accelerator === 'cpu') return CPU_DEVICE;
    return CPU_DEVICE;
  }

  async _prepareDevice(device) {
    if (device.type !== 'gpu') return;
    const { preloadCudaRuntime } = await import('../../compiler/runtime/backend_registry.js');
    await preloadCudaRuntime();
  }

  _resolveLoggers(loggerConfig) {
    if (loggerConfig === false || loggerConfig === null) return [];
    if (loggerConfig === true) return [new ConsoleLogger()];
    if (Array.isArray(loggerConfig)) return loggerConfig;
    return [loggerConfig];
  }

  _extractHyperparams(model, optimizers) {
    const params = {
      maxEpochs: this._state.maxEpochs,
      maxSteps: this._state.maxSteps,
      accelerator: this._accelerator,
      precision: this._precision,
      accumulateGradBatches: this._accumulateGradBatches,
    };
    for (let i = 0; i < optimizers.length; i++) {
      const opt = optimizers[i];
      const defaults = opt.defaults;
      const prefix = optimizers.length > 1 ? `optimizer_${i}_` : '';
      params[prefix + 'optimizer'] = opt.constructor.name;
      if (defaults.lr !== undefined) params[prefix + 'lr'] = defaults.lr;
      if (defaults.weightDecay !== undefined) params[prefix + 'weight_decay'] = defaults.weightDecay;
    }
    return params;
  }
}
