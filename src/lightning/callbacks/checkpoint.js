import { fs } from '#io/fs';
import { joinPath } from '../../io/path.js';
import { Callback } from './callback.js';
import { serializeCheckpoint, deserializeCheckpoint } from '../io/checkpoint_format.js';

const CKPT_EXT = '.ckpt';

export class ModelCheckpoint extends Callback {
  constructor({
    dirpath = './lightning_logs/checkpoints',
    filename = 'epoch={epoch}-step={step}',
    monitor = null,
    mode = 'min',
    saveTopK = 1,
    saveLast = true,
    everyNEpochs = 1,
  } = {}) {
    super();
    this._dirpath = dirpath;
    this._filename = filename;
    this._monitor = monitor;
    this._mode = mode;
    this._saveTopK = saveTopK;
    this._saveLast = saveLast;
    this._everyNEpochs = everyNEpochs;
    this._bestK = [];
    this._compareFn = mode === 'min'
      ? (a, b) => a - b
      : (a, b) => b - a;
    this._bestModelPath = null;
    this._lastModelPath = null;
  }

  get bestModelPath() { return this._bestModelPath; }
  get lastModelPath() { return this._lastModelPath; }
  get bestKModels() { return this._bestK; }

  onTrainEpochEnd(trainer, model) {
    const state = trainer.state;
    if ((state.epoch + 1) % this._everyNEpochs !== 0) return;

    this._ensureDir();
    const filledName = this._fillTemplate(state);

    if (this._saveLast) {
      const lastPath = joinPath(this._dirpath, 'last' + CKPT_EXT);
      this._saveCheckpoint(model, trainer, lastPath);
      this._lastModelPath = lastPath;
    }

    if (!this._monitor) {
      const path = joinPath(this._dirpath, filledName + CKPT_EXT);
      this._saveCheckpoint(model, trainer, path);
      return;
    }

    const metrics = state.epochMetrics.computeAll();
    const current = metrics[this._monitor];
    if (current === undefined) return;

    const path = joinPath(this._dirpath, filledName + CKPT_EXT);
    const entry = { score: current, path };

    if (this._saveTopK < 0) {
      this._saveCheckpoint(model, trainer, path);
      this._bestK.push(entry);
      this._updateBest();
      return;
    }

    const insertIdx = this._findInsertIndex(current);
    if (insertIdx < this._saveTopK) {
      this._saveCheckpoint(model, trainer, path);
      this._bestK.splice(insertIdx, 0, entry);
      if (this._bestK.length > this._saveTopK) {
        const removed = this._bestK.pop();
        this._tryDelete(removed.path);
      }
      this._updateBest();
    }
  }

  _findInsertIndex(score) {
    let lo = 0;
    let hi = this._bestK.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this._compareFn(score, this._bestK[mid].score) < 0) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  _updateBest() {
    if (this._bestK.length > 0) {
      this._bestModelPath = this._bestK[0].path;
    }
  }

  _saveCheckpoint(model, trainer, path) {
    const checkpoint = {
      epoch: trainer.state.epoch,
      globalStep: trainer.state.globalStep,
      modelState: model.stateDict(),
    };

    const optimizers = model._currentOptimizers;
    if (optimizers && optimizers.length > 0) {
      checkpoint.optimizerStates = optimizers.map((opt) => opt.stateDict());
    }

    trainer.callbackConnector.dispatch('onSaveCheckpoint', trainer, model, checkpoint);

    const tmp = path + '.tmp';
    fs.writeBinary(tmp, serializeCheckpoint(checkpoint));
    fs.rename(tmp, path);
  }

  _fillTemplate(state) {
    return this._filename
      .replace('{epoch}', state.epoch)
      .replace('{step}', state.globalStep);
  }

  _ensureDir() {
    if (!fs.exists(this._dirpath)) {
      fs.mkdir(this._dirpath);
    }
  }

  _tryDelete(path) {
    try { fs.remove(path); } catch { /* noop */ }
  }
}

export function loadCheckpoint(path) {
  return deserializeCheckpoint(fs.readBinary(path));
}

export function applyCheckpoint(checkpoint, model, optimizers = []) {
  if (checkpoint.modelState) model.loadStateDict(checkpoint.modelState);
  if (checkpoint.optimizerStates) {
    const n = Math.min(optimizers.length, checkpoint.optimizerStates.length);
    for (let i = 0; i < n; i++) optimizers[i].loadStateDict(checkpoint.optimizerStates[i]);
  }
  return checkpoint;
}
