import { Callback } from './callback.js';

export class LearningRateMonitor extends Callback {
  constructor({ logMomentum = false } = {}) {
    super();
    this._logMomentum = logMomentum;
    this._lrHistory = {};
  }

  get lrHistory() {
    return this._lrHistory;
  }

  onTrainBatchStart(trainer, model, _batch, _batchIdx) {
    const optimizers = model._currentOptimizers;
    if (!optimizers) return;

    for (let oi = 0; oi < optimizers.length; oi++) {
      const groups = optimizers[oi].paramGroups;
      for (let gi = 0; gi < groups.length; gi++) {
        const name = optimizers.length > 1 || groups.length > 1
          ? `lr_opt${oi}_group${gi}`
          : 'lr';
        const lr = groups[gi].lr;
        model.log(name, lr, { onStep: true, onEpoch: false, progBar: false });

        if (!this._lrHistory[name]) this._lrHistory[name] = [];
        this._lrHistory[name].push({ step: trainer.state.globalStep, lr });

        if (this._logMomentum && groups[gi].momentum !== undefined) {
          const mName = name.replace('lr', 'momentum');
          model.log(mName, groups[gi].momentum, { onStep: true, onEpoch: false });
        }
      }
    }
  }
}
