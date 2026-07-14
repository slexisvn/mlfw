import { TrainerState, SingleDeviceStrategy, Stage } from './state.js';
import { CallbackConnector, LoggerConnector } from './hooks.js';
import { FitLoop } from './loops/fit_loop.js';
import { EvaluationLoop } from './loops/evaluation_loop.js';
import { PredictionLoop } from './loops/prediction_loop.js';
import { parseOptimizersConfig } from './module.js';
import { ConsoleLogger } from '../loggers/console.js';
import { ProgressCallback } from '../callbacks/progress.js';
import { ModelCheckpoint } from '../callbacks/checkpoint.js';
import { CPU_DEVICE, GPU_DEVICE, WASM_DEVICE, WEBGPU_DEVICE } from '../../tensor/types/device.js';
import type { Device } from '../../tensor/types/device.js';
import type { Callback } from '../callbacks/callback.js';
import type { Logger } from '../loggers/logger.js';
import type {
  DataLoaderLike,
  HyperparameterRecord,
  LightningModuleLike,
  NumericMetricRecord,
  OptimizerLike,
  TrainerOptions,
} from '../types.js';
import type { SchedulerConfig } from './module.js';

export class Trainer {
  private _state: TrainerState;
  private _compile: boolean;
  private _compileMode: string;
  private _cudaGraph: boolean;
  private _cudaGraphWarmupSteps: number;
  private _accelerator: TrainerOptions['accelerator'];
  private _precision: string;
  private _gradientClipVal: number | null;
  private _gradientClipAlgorithm: string;
  private _accumulateGradBatches: number;
  private _limitTrainBatches: number | null;
  private _limitValBatches: number | null;
  private _limitTestBatches: number | null;
  private _valCheckInterval: number;
  private _checkValEveryNEpoch: number;
  private _logEveryNSteps: number;
  private _deterministic: boolean;
  private _defaultRootDir: string;
  private _loggers: Logger[];
  private _strategy: SingleDeviceStrategy;
  private _fitLoop: FitLoop;
  private _evaluationLoop: EvaluationLoop;
  private _predictionLoop: PredictionLoop;
  private _callbackConnector: CallbackConnector;
  private _loggerConnector: LoggerConnector;
  private _model: LightningModuleLike | null;
  private _webgpuMod: { flushWebGPUEager(): Promise<void> } | null;

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
    cudaGraph = false,
    cudaGraphWarmupSteps = 3,
  }: TrainerOptions = {}) {
    this._state = new TrainerState();
    this._state.maxEpochs = maxEpochs;
    this._state.maxSteps = maxSteps;

    this._compile = compile;
    this._compileMode = compileMode;
    this._cudaGraph = cudaGraph;
    this._cudaGraphWarmupSteps = cudaGraphWarmupSteps;
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
    this._webgpuMod = null;
  }

  get state(): TrainerState { return this._state; }
  get strategy(): SingleDeviceStrategy { return this._strategy; }
  get callbackConnector(): CallbackConnector { return this._callbackConnector; }
  get loggerConnector(): LoggerConnector { return this._loggerConnector; }
  get fitLoop(): FitLoop { return this._fitLoop; }
  get gradientClipVal(): number | null { return this._gradientClipVal; }
  get gradientClipAlgorithm(): string { return this._gradientClipAlgorithm; }
  get compile(): boolean { return this._compile; }
  get compileMode(): string { return this._compileMode; }
  get cudaGraph(): boolean { return this._cudaGraph; }
  get cudaGraphWarmupSteps(): number { return this._cudaGraphWarmupSteps; }
  get accumulateGradBatches(): number { return this._accumulateGradBatches; }
  set accumulateGradBatches(v: number) { this._accumulateGradBatches = v; }
  get limitTrainBatches(): number | null { return this._limitTrainBatches; }
  get limitValBatches(): number | null { return this._limitValBatches; }
  get limitTestBatches(): number | null { return this._limitTestBatches; }
  get checkValEveryNEpoch(): number { return this._checkValEveryNEpoch; }
  get logEveryNSteps(): number { return this._logEveryNSteps; }
  get shouldStop(): boolean { return this._state.shouldStop; }
  set shouldStop(v: boolean) { this._state.shouldStop = v; }
  get currentEpoch(): number { return this._state.epoch; }
  get globalStep(): number { return this._state.globalStep; }
  get logger(): Logger | null { return this._loggers[0] || null; }
  get loggers(): Logger[] { return this._loggers; }
  get callbacks(): Callback[] { return this._callbackConnector.callbacks; }
  get model(): LightningModuleLike | null { return this._model; }
  get defaultRootDir(): string { return this._defaultRootDir; }

  async fit(model: LightningModuleLike, trainLoader: DataLoaderLike, valLoader: DataLoaderLike | null = null): Promise<void> {
    this._model = model;
    model._trainer = this;
    const device = this._resolveDevice();
    this._guardEagerWebGPU(device, 'fit', valLoader != null);
    this._guardCudaGraph(device, valLoader != null);
    model._device = device;
    await this._prepareDevice(device);
    this._strategy.setup(model, device);

    const { optimizers, schedulerConfigs } = parseOptimizersConfig(
      await Promise.resolve(model.configureOptimizers())
    );
    if (this._cudaGraph && schedulerConfigs && schedulerConfigs.some((c) => c && c.scheduler)) {
      throw new Error('Trainer(cudaGraph=true) v1 requires a constant learning rate: LR schedulers change lr, but lr is baked into the captured graph. Remove the scheduler or disable cudaGraph.');
    }
    model._currentOptimizers = optimizers;

    this._loggerConnector.logHyperparams(this._extractHyperparams(model, optimizers));

    this._callbackConnector.dispatch('setup', this, model, Stage.TRAINING);
    this._callbackConnector.dispatch('onFitStart', this, model);

    this._state.shouldStop = false;
    await this._fitLoop.run(model, trainLoader, valLoader, this, optimizers, schedulerConfigs);

    if (device === GPU_DEVICE) {
      const { teardownAfterFit } = await import('#io/cuda_runtime');
      teardownAfterFit(model, optimizers);
    }

    this._callbackConnector.dispatch('onFitEnd', this, model);
    this._callbackConnector.dispatch('teardown', this, model, Stage.TRAINING);

    for (let i = 0; i < this._loggers.length; i++) {
      this._loggers[i].finalize();
    }
  }

  async validate(model: LightningModuleLike, dataLoader: DataLoaderLike): Promise<NumericMetricRecord> {
    model._trainer = this;
    this._model = model;
    const device = this._resolveDevice();
    this._guardEagerWebGPU(device, 'validate');
    model._device = device;
    await this._prepareDevice(device);
    this._strategy.setup(model, device);
    this._callbackConnector.dispatch('setup', this, model, Stage.VALIDATING);
    const metrics = await this._fitLoop.validationLoop.run(model, dataLoader, this, null);
    this._callbackConnector.dispatch('teardown', this, model, Stage.VALIDATING);
    for (let i = 0; i < this._loggers.length; i++) this._loggers[i].finalize();
    return metrics;
  }

  async test(model: LightningModuleLike, dataLoader: DataLoaderLike): Promise<NumericMetricRecord> {
    model._trainer = this;
    this._model = model;
    const device = this._resolveDevice();
    this._guardEagerWebGPU(device, 'test');
    model._device = device;
    await this._prepareDevice(device);
    this._strategy.setup(model, device);
    this._callbackConnector.dispatch('setup', this, model, Stage.TESTING);
    const metrics = await this._evaluationLoop.run(model, dataLoader, this);
    this._callbackConnector.dispatch('teardown', this, model, Stage.TESTING);
    for (let i = 0; i < this._loggers.length; i++) this._loggers[i].finalize();
    return metrics;
  }

  async predict(model: LightningModuleLike, dataLoader: DataLoaderLike): Promise<unknown[]> {
    model._trainer = this;
    this._model = model;
    const device = this._resolveDevice();
    model._device = device;
    await this._prepareDevice(device);
    this._strategy.setup(model, device);
    return await this._predictionLoop.run(model, dataLoader, this);
  }

  _resolveDevice(): Device {
    if (this._accelerator === 'gpu') return GPU_DEVICE;
    if (this._accelerator === 'wasm') return WASM_DEVICE;
    if (this._accelerator === 'webgpu') return WEBGPU_DEVICE;
    if (this._accelerator === 'cpu') return CPU_DEVICE;
    return CPU_DEVICE;
  }

  _guardEagerWebGPU(device: Device, stage: string, hasValidation = false): void {
    if (device.type !== 'webgpu') return;
    if (stage === 'fit') {
      if (!this._compile) {
        throw new Error('Trainer(accelerator="webgpu"): eager WebGPU is inference-only (CUSTOM_0 dispatch has no autograd key). Pass compile=true to train on WebGPU, or use predict() for eager inference.');
      }
      if (hasValidation) {
        throw new Error('Trainer(accelerator="webgpu"): in-fit validation is unsupported — validationStep runs eagerly and reads scalar metrics via .item(), which WebGPU\'s asynchronous readback cannot serve, and there is no compiled validation path. Call fit() without a valLoader on WebGPU.');
      }
      return;
    }
    throw new Error(`Trainer(accelerator="webgpu"): ${stage}() reads scalar metrics synchronously via .item(), which WebGPU's asynchronous readback cannot serve eagerly. Use predict() for eager WebGPU inference, or train via compile=true.`);
  }

  _guardCudaGraph(device: Device, hasValidation = false): void {
    if (!this._cudaGraph) return;
    if (device.type !== 'gpu') {
      throw new Error('Trainer(cudaGraph=true) requires accelerator="gpu" (eager CUDA whole-step capture/replay).');
    }
    if (this._compile) {
      throw new Error('Trainer(cudaGraph=true) is incompatible with compile=true: CUDA graph capture targets the eager training step, not the compiled path.');
    }
    if (this._gradientClipVal != null && this._gradientClipAlgorithm !== 'norm') {
      throw new Error('Trainer(cudaGraph=true) supports gradient_clip_algorithm="norm" only; "value" clipping is not yet device-side.');
    }
    if (this._accumulateGradBatches !== 1) {
      throw new Error('Trainer(cudaGraph=true) v1 requires accumulateGradBatches=1.');
    }
    if (hasValidation) {
      throw new Error('Trainer(cudaGraph=true) v1 does not support in-fit validation. Call fit() without a valLoader.');
    }
  }

  async _prepareDevice(device: Device): Promise<void> {
    if (device.type === 'gpu') {
      const { preloadCudaRuntime } = await import('../../runtime/backend_registry.js');
      await preloadCudaRuntime();
    } else if (device.type === 'webgpu') {
      const { preloadWebGPU } = await import('../../runtime/backend_registry.js');
      this._webgpuMod = await preloadWebGPU() as { flushWebGPUEager(): Promise<void> };
    }
  }

  async _flushEagerInference(): Promise<void> {
    if (this._webgpuMod) await this._webgpuMod.flushWebGPUEager();
  }

  _resolveLoggers(loggerConfig: TrainerOptions['logger']): Logger[] {
    if (loggerConfig === false || loggerConfig === null) return [];
    if (loggerConfig === true) return [new ConsoleLogger()];
    if (Array.isArray(loggerConfig)) return loggerConfig;
    return [loggerConfig as Logger];
  }

  _extractHyperparams(model: LightningModuleLike, optimizers: OptimizerLike[]): HyperparameterRecord {
    const params: HyperparameterRecord = {
      maxEpochs: this._state.maxEpochs,
      maxSteps: this._state.maxSteps,
      accelerator: this._accelerator,
      precision: this._precision,
      accumulateGradBatches: this._accumulateGradBatches,
    };
    for (let i = 0; i < optimizers.length; i++) {
      const opt = optimizers[i];
      const defaults = opt.defaults || {};
      const prefix = optimizers.length > 1 ? `optimizer_${i}_` : '';
      params[prefix + 'optimizer'] = opt.constructor.name;
      if (defaults.lr !== undefined) params[prefix + 'lr'] = defaults.lr;
      if (defaults.weightDecay !== undefined) params[prefix + 'weight_decay'] = defaults.weightDecay;
    }
    return params;
  }
}
