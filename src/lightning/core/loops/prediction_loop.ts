import { Stage } from '../state.js';
import { resolveLimit } from './utils.js';
import { noGrad } from '../../../autograd/grad_mode.js';
import type { DataLoaderLike, LightningModuleLike, TrainerCoreLike } from '../../types.js';

export class PredictionLoop {
  async run(model: LightningModuleLike, dataLoader: DataLoaderLike, trainer: TrainerCoreLike): Promise<unknown[]> {
    const state = trainer.state;
    const callbacks = trainer.callbackConnector;
    const strategy = trainer.strategy;
    const prevStage = state.stage;
    state.stage = Stage.PREDICTING;

    model.eval();
    callbacks.dispatch('onPredictStart', trainer, model);

    const predictions: unknown[] = [];
    const limit = resolveLimit(trainer.limitTestBatches, dataLoader.length);
    let batchIdx = 0;

    await noGrad(async () => {
      for (const rawBatch of dataLoader) {
        if (batchIdx >= limit) break;
        const batch = strategy.toDevice(rawBatch);
        callbacks.dispatch('onPredictBatchStart', trainer, model, batch, batchIdx);
        const output = await Promise.resolve(model.predictStep(batch, batchIdx));
        await trainer._flushEagerInference();
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
