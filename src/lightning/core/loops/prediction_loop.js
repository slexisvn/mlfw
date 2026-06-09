import { Stage } from '../state.js';

export class PredictionLoop {
  async run(model, dataLoader, trainer) {
    const state = trainer.state;
    const callbacks = trainer.callbackConnector;
    const prevStage = state.stage;
    state.stage = Stage.PREDICTING;

    model.eval();
    callbacks.dispatch('onPredictStart', trainer, model);

    const predictions = [];
    const limit = resolveLimit(trainer.limitTestBatches, dataLoader.length);
    let batchIdx = 0;

    await noGradAsync(async () => {
      for (const batch of dataLoader) {
        if (batchIdx >= limit) break;
        callbacks.dispatch('onPredictBatchStart', trainer, model, batch, batchIdx);
        const output = await Promise.resolve(model.predictStep(batch, batchIdx));
        predictions.push(output);
        callbacks.dispatch('onPredictBatchEnd', trainer, model, output, batch, batchIdx);
        batchIdx++;
      }
    });

    callbacks.dispatch('onPredictEnd', trainer, model);
    model.train();
    state.stage = prevStage;
    return predictions;
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
