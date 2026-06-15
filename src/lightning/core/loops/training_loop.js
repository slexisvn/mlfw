import { Stage } from '../state.js';
import { clipGradNorm_, clipGradValue_ } from '../../../optim/utils.js';
import { div } from '../../../tensor/ops/ops.js';
import { eagerFlush } from '../../../dispatcher/eager_mode.js';

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

    for (const rawBatch of dataLoader) {
      if (batchIdx >= limit) break;
      if (state.shouldStop) break;
      if (state.maxSteps > 0 && state.globalStep >= state.maxSteps) {
        state.shouldStop = true;
        break;
      }

      const batch = strategy.toDevice(rawBatch);
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
      eagerFlush();
      state.globalStep++;
      batchIdx++;
    }

    eagerFlush();

    const epochMetrics = loggerConnector.flushEpochMetrics(state.globalStep);
    this._stepEpochSchedulers(schedulerConfigs, state.epoch);
    model.onTrainEpochEnd();
    callbacks.dispatch('onTrainEpochEnd', trainer, model);
    return epochMetrics;
  }

  async _automaticStep(model, batch, batchIdx, trainer, optimizers, schedulerConfigs, strategy, accumGrad, callbacks) {
    if (trainer.compile) {
      return this._compiledStep(model, batch, batchIdx, trainer, optimizers, schedulerConfigs, accumGrad);
    }
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

  async _compiledStep(model, batch, batchIdx, trainer, optimizers, schedulerConfigs, accumGrad) {
    const elems = Array.isArray(batch) ? batch : [batch];
    const callForward = (...xs) => model.trainingStep(Array.isArray(batch) ? xs : xs[0], 0);
    let loss;
    if (!model.__compiledTrainStep) {
      const { compileWithBackward } = await import('../../../tracing/compile_backward.js');
      const { CPUTarget, CUDATarget } = await import('../../../backend/target.js');
      const target = model._device && model._device.type === 'gpu' ? CUDATarget() : CPUTarget();
      model.__compiledTrainStep = compileWithBackward({ forward: callForward }, elems, { target, mode: trainer.compileMode });
      const origLog = model.log;
      model.log = () => {};
      try { loss = model.__compiledTrainStep(...elems); if (loss && loss.then) loss = await loss; }
      finally { model.log = origLog; }
    } else {
      loss = model.__compiledTrainStep(...elems); if (loss && loss.then) loss = await loss;
    }

    const step = model.__compiledTrainStep;
    const params = step.capturedParams();
    const { ones } = await import('../../../tensor/factory/creation_ops.js');
    let grads = step.backward(ones(loss.shape)); if (grads && grads.then) grads = await grads;
    const off = grads.length - params.length;
    for (let i = 0; i < params.length; i++) { const g = grads[off + i]; if (g) params[i].grad = g; }

    if ((batchIdx + 1) % accumGrad === 0) {
      for (let i = 0; i < optimizers.length; i++) {
        this._clipGradients(model, trainer);
        optimizers[i].step();
        optimizers[i].zeroGrad();
      }
      this._stepStepSchedulers(schedulerConfigs, trainer.state.globalStep);
    }

    if (model.log) model.log('train_loss', loss);
    return loss;
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
