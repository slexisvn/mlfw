import { TrainingLoop } from './training_loop.js';
import { ValidationLoop } from './validation_loop.js';
import type { DataLoaderLike, LightningModuleLike, OptimizerLike, TrainerCoreLike } from '../../types.js';
import type { SchedulerConfig } from '../module.js';

export class FitLoop {
  private _trainingLoop: TrainingLoop;
  private _validationLoop: ValidationLoop;

  constructor() {
    this._trainingLoop = new TrainingLoop();
    this._validationLoop = new ValidationLoop();
  }

  get trainingLoop(): TrainingLoop {
    return this._trainingLoop;
  }

  get validationLoop(): ValidationLoop {
    return this._validationLoop;
  }

  async run(
    model: LightningModuleLike,
    trainLoader: DataLoaderLike,
    valLoader: DataLoaderLike | null,
    trainer: TrainerCoreLike,
    optimizers: OptimizerLike[],
    schedulerConfigs: Array<SchedulerConfig | null>
  ): Promise<void> {
    const state = trainer.state;
    const callbacks = trainer.callbackConnector;

    callbacks.dispatch('onTrainStart', trainer, model);

    for (let epoch = 0; epoch < state.maxEpochs; epoch++) {
      if (state.shouldStop) break;
      if (state.maxSteps > 0 && state.globalStep >= state.maxSteps) break;

      state.epoch = epoch;
      state.resetEpochMetrics();

      await this._trainingLoop.run(model, trainLoader, trainer, optimizers, schedulerConfigs);

      if (state.shouldStop) break;

      const shouldValidate = valLoader && this._shouldRunValidation(epoch, trainer);
      if (shouldValidate) {
        await this._validationLoop.run(model, valLoader, trainer, schedulerConfigs);
      }
    }

    callbacks.dispatch('onTrainEnd', trainer, model);
  }

  _shouldRunValidation(epoch: number, trainer: TrainerCoreLike): boolean {
    const interval = trainer.checkValEveryNEpoch;
    return (epoch + 1) % interval === 0;
  }
}
