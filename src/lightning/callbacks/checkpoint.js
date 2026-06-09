import { writeFileSync, existsSync, mkdirSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { Callback } from './callback.js';

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
      const lastPath = join(this._dirpath, 'last.ckpt.json');
      this._saveCheckpoint(model, trainer, lastPath);
      this._lastModelPath = lastPath;
    }

    if (!this._monitor) {
      const path = join(this._dirpath, filledName + '.ckpt.json');
      this._saveCheckpoint(model, trainer, path);
      return;
    }

    const metrics = state.epochMetrics.computeAll();
    const current = metrics[this._monitor];
    if (current === undefined) return;

    const path = join(this._dirpath, filledName + '.ckpt.json');
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
      modelState: serializeStateDict(model.stateDict()),
    };

    const optimizers = model._currentOptimizers;
    if (optimizers && optimizers.length > 0) {
      checkpoint.optimizerStates = [];
      for (let i = 0; i < optimizers.length; i++) {
        checkpoint.optimizerStates.push(serializeOptimizerState(optimizers[i].stateDict()));
      }
    }

    trainer.callbackConnector.dispatch('onSaveCheckpoint', trainer, model, checkpoint);
    writeFileSync(path, JSON.stringify(checkpoint));
  }

  _fillTemplate(state) {
    return this._filename
      .replace('{epoch}', state.epoch)
      .replace('{step}', state.globalStep);
  }

  _ensureDir() {
    if (!existsSync(this._dirpath)) {
      mkdirSync(this._dirpath, { recursive: true });
    }
  }

  _tryDelete(path) {
    try { unlinkSync(path); } catch { /* noop */ }
  }
}

function serializeStateDict(stateDict) {
  const result = {};
  for (const [key, tensor] of stateDict) {
    const data = tensor._impl.storage.data;
    result[key] = {
      shape: tensor.shape,
      dtype: tensor.dtype,
      data: encodeTypedArray(data),
    };
  }
  return result;
}

function serializeOptimizerState(stateDict) {
  const serialized = {
    paramGroups: stateDict.paramGroups,
    state: {},
  };
  for (const [id, s] of stateDict.state) {
    const cloned = {};
    for (const key of Object.keys(s)) {
      const val = s[key];
      if (ArrayBuffer.isView(val)) {
        cloned[key] = { type: val.constructor.name, data: encodeTypedArray(val) };
      } else {
        cloned[key] = val;
      }
    }
    serialized.state[id] = cloned;
  }
  return serialized;
}

function encodeTypedArray(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
