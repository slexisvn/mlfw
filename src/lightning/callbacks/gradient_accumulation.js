import { Callback } from './callback.js';

export class GradientAccumulationScheduler extends Callback {
  constructor({ scheduling }) {
    super();
    this._scheduling = new Map();
    const entries = Object.entries(scheduling);
    for (let i = 0; i < entries.length; i++) {
      this._scheduling.set(Number(entries[i][0]), entries[i][1]);
    }
    this._sortedEpochs = [...this._scheduling.keys()].sort((a, b) => a - b);
  }

  onTrainEpochStart(trainer, _model) {
    const epoch = trainer.state.epoch;
    if (this._scheduling.has(epoch)) {
      trainer.accumulateGradBatches = this._scheduling.get(epoch);
    }
  }

  getCurrentAccumulation(epoch) {
    let result = 1;
    for (let i = 0; i < this._sortedEpochs.length; i++) {
      if (this._sortedEpochs[i] <= epoch) {
        result = this._scheduling.get(this._sortedEpochs[i]);
      } else {
        break;
      }
    }
    return result;
  }
}
