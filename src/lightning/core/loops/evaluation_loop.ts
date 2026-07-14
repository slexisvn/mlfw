import { Stage } from '../state.js';
import { resolveLimit, noGradAsync } from './utils.js';
import type { DataLoaderLike, LightningModuleLike, NumericMetricRecord, TrainerCoreLike } from '../../types.js';

export class EvaluationLoop {
  async run(model: LightningModuleLike, dataLoader: DataLoaderLike, trainer: TrainerCoreLike): Promise<NumericMetricRecord> {
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
