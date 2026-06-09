import { Stage } from '../state.js';
import { clipGradNorm_, clipGradValue_ } from '../../../optim/utils.js';
import { div } from '../../../tensor/ops/ops.js';

export class TrainingLoop {
  async run(model, dataLoader, trainer, optimizers, schedulerConfigs) {
    const state = trainer.state;
    const callbacks = trainer.callbackConnector;
    const loggerConnector = trainer.loggerConnector;
    const strategy = trainer.strategy;
    const accumGrad = trainer.accumulateGradBatches;
    const limit = resolveLimit(trainer.limitTrainBatches, dataLoader.length);

    state.stage = Stage.TRAINING;
    model.train();
    model.onTrainEpochStart();
    callbacks.dispatch('onTrainEpochStart', trainer, model);

    let batchIdx = 0;

    for (const batch of dataLoader) {
      if (batchIdx >= limit) break;
      if (state.shouldStop) break;
      if (state.maxSteps > 0 && state.globalStep >= state.maxSteps) {
        state.shouldStop = true;
        break;
      }

      callbacks.dispatch('onTrainBatchStart', trainer, model, batch, batchIdx);

      let output;
      if (model.automaticOptimization) {
        output = await this._automaticStep(
          model, batch, batchIdx, trainer, optimizers,
          schedulerConfigs, strategy, accumGrad, callbacks
        );
      } else {
        output = await Promise.resolve(model.trainingStep(batch, batchIdx));
      }

      loggerConnector.drain(model);

      if (state.globalStep % trainer.logEveryNSteps === 0) {
        loggerConnector.flushStepMetrics(state.globalStep);
      }

      callbacks.dispatch('onTrainBatchEnd', trainer, model, output, batch, batchIdx);
      state.globalStep++;
      batchIdx++;
    }

    const epochMetrics = loggerConnector.flushEpochMetrics(state.globalStep);
    this._stepEpochSchedulers(schedulerConfigs, state.epoch);
    model.onTrainEpochEnd();
    callbacks.dispatch('onTrainEpochEnd', trainer, model);
    return epochMetrics;
  }

  async _automaticStep(model, batch, batchIdx, trainer, optimizers, schedulerConfigs, strategy, accumGrad, callbacks) {
    const result = await Promise.resolve(model.trainingStep(batch, batchIdx));
    let loss = result;
    let output = result;
    if (result && typeof result === 'object' && !(result.backward)) {
      loss = result.loss;
      output = result;
    }

    if (accumGrad > 1) {
      loss = div(loss, accumGrad);
    }

    callbacks.dispatch('onBeforeBackward', trainer, model, loss);
    strategy.backward(loss);
    callbacks.dispatch('onAfterBackward', trainer, model);

    const isAccumBoundary = (batchIdx + 1) % accumGrad === 0;
    if (isAccumBoundary) {
      for (let i = 0; i < optimizers.length; i++) {
        this._clipGradients(model, trainer);
        callbacks.dispatch('onBeforeOptimizerStep', trainer, model, optimizers[i]);
        strategy.optimizerStep(optimizers[i]);
        callbacks.dispatch('onBeforeZeroGrad', trainer, model, optimizers[i]);
        optimizers[i].zeroGrad();
      }
      this._stepStepSchedulers(schedulerConfigs, trainer.state.globalStep);
    }

    return output;
  }

  _clipGradients(model, trainer) {
    if (!trainer.gradientClipVal) return;
    const params = [...model.parameters()];
    if (trainer.gradientClipAlgorithm === 'norm') {
      clipGradNorm_(params, trainer.gradientClipVal);
    } else {
      clipGradValue_(params, trainer.gradientClipVal);
    }
  }

  _stepStepSchedulers(schedulerConfigs, globalStep) {
    if (!schedulerConfigs) return;
    for (let i = 0; i < schedulerConfigs.length; i++) {
      const cfg = schedulerConfigs[i];
      if (!cfg || cfg.interval !== 'step') continue;
      if (cfg.monitor) continue;
      if ((globalStep + 1) % cfg.frequency === 0) {
        cfg.scheduler.step();
      }
    }
  }

  _stepEpochSchedulers(schedulerConfigs, epoch) {
    if (!schedulerConfigs) return;
    for (let i = 0; i < schedulerConfigs.length; i++) {
      const cfg = schedulerConfigs[i];
      if (!cfg || cfg.interval !== 'epoch') continue;
      if (cfg.monitor) continue;
      if ((epoch + 1) % cfg.frequency === 0) {
        cfg.scheduler.step();
      }
    }
  }
}

function resolveLimit(limitConfig, totalBatches) {
  if (limitConfig === null || limitConfig === undefined) return totalBatches;
  if (typeof limitConfig === 'number') {
    if (limitConfig > 0 && limitConfig <= 1) {
      return Math.max(1, Math.round(limitConfig * totalBatches));
    }
    return Math.min(limitConfig, totalBatches);
  }
  return totalBatches;
}
