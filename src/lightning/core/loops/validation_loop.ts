
import { Stage } from '../state.js';
import { resolveLimit } from './utils.js';
import { noGrad } from '../../../autograd/grad_mode.js';
import type { DataLoaderLike, LightningModuleLike, NumericMetricRecord, TrainerCoreLike } from '../../types.js';
import type { SchedulerConfig } from '../module.js';

export class ValidationLoop {
  async run(
    model: LightningModuleLike,
    dataLoader: DataLoaderLike,
    trainer: TrainerCoreLike,
    schedulerConfigs: Array<SchedulerConfig | null> | null
  ): Promise<NumericMetricRecord> {
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
    state.numValBatches = limit;
    let batchIdx = 0;

    await noGrad(async () => {
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

  private _stepPlateauSchedulers(schedulerConfigs: Array<SchedulerConfig | null> | null, metrics: NumericMetricRecord): void {
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
