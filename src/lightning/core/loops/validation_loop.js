import { noGrad } from '../../../autograd/grad_mode.js';
import { Stage } from '../state.js';

export class ValidationLoop {
  async run(model, dataLoader, trainer, schedulerConfigs) {
    const state = trainer.state;
    const callbacks = trainer.callbackConnector;
    const loggerConnector = trainer.loggerConnector;
    const prevStage = state.stage;
    state.stage = Stage.VALIDATING;

    model.eval();
    model.onValidationEpochStart();
    callbacks.dispatch('onValidationStart', trainer, model);
    callbacks.dispatch('onValidationEpochStart', trainer, model);

    const limit = resolveLimit(trainer.limitValBatches, dataLoader.length);
    let batchIdx = 0;

    await noGradAsync(async () => {
      for (const batch of dataLoader) {
        if (batchIdx >= limit) break;
        callbacks.dispatch('onValidationBatchStart', trainer, model, batch, batchIdx);
        const output = await Promise.resolve(model.validationStep(batch, batchIdx));
        loggerConnector.drain(model);
        callbacks.dispatch('onValidationBatchEnd', trainer, model, output, batch, batchIdx);
        batchIdx++;
      }
    });

    const metrics = loggerConnector.flushEpochMetrics(state.globalStep);
    this._stepPlateauSchedulers(schedulerConfigs, metrics);
    model.onValidationEpochEnd();
    callbacks.dispatch('onValidationEpochEnd', trainer, model);
    callbacks.dispatch('onValidationEnd', trainer, model);
    model.train();
    state.stage = prevStage;
    return metrics;
  }

  _stepPlateauSchedulers(schedulerConfigs, metrics) {
    if (!schedulerConfigs) return;
    for (let i = 0; i < schedulerConfigs.length; i++) {
      const cfg = schedulerConfigs[i];
      if (!cfg || !cfg.monitor) continue;
      const scheduler = cfg.scheduler;
      if (typeof scheduler.step === 'function' && scheduler.step.length > 0) {
        const metricVal = metrics[cfg.monitor];
        if (metricVal !== undefined) {
          scheduler.step(metricVal);
        }
      }
    }
  }
}

async function noGradAsync(fn) {
  const { GradMode } = await import('../../../autograd/grad_mode.js');
  const prev = GradMode.isEnabled();
  GradMode.setEnabled(false);
  try {
    await fn();
  } finally {
    GradMode.setEnabled(prev);
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
