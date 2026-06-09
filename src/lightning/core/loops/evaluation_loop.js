import { Stage } from '../state.js';

export class EvaluationLoop {
  async run(model, dataLoader, trainer) {
    const state = trainer.state;
    const callbacks = trainer.callbackConnector;
    const loggerConnector = trainer.loggerConnector;
    const prevStage = state.stage;
    state.stage = Stage.TESTING;

    model.eval();
    model.onTestEpochStart();
    callbacks.dispatch('onTestStart', trainer, model);
    callbacks.dispatch('onTestEpochStart', trainer, model);

    const limit = resolveLimit(trainer.limitTestBatches, dataLoader.length);
    let batchIdx = 0;

    await noGradAsync(async () => {
      for (const batch of dataLoader) {
        if (batchIdx >= limit) break;
        callbacks.dispatch('onTestBatchStart', trainer, model, batch, batchIdx);
        const output = await Promise.resolve(model.testStep(batch, batchIdx));
        loggerConnector.drain(model);
        callbacks.dispatch('onTestBatchEnd', trainer, model, output, batch, batchIdx);
        batchIdx++;
      }
    });

    const metrics = loggerConnector.flushEpochMetrics(state.globalStep);
    model.onTestEpochEnd();
    callbacks.dispatch('onTestEpochEnd', trainer, model);
    callbacks.dispatch('onTestEnd', trainer, model);
    model.train();
    state.stage = prevStage;
    return metrics;
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
