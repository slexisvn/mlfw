import { fs } from '#io/fs';
import { joinPath } from '../../runtime/path.js';
import { Callback } from './callback.js';
import { serializeCheckpoint, deserializeCheckpoint } from '../io/checkpoint_format.js';
import type { LightningModuleLike, TrainerLike } from '../types.js';

const CKPT_EXT = '.ckpt';

type CheckpointMode = 'min' | 'max';
type BestKEntry = {
  score: number;
  path: string;
};
type ModelCheckpointOptions = {
  dirpath?: string;
  filename?: string;
  monitor?: string | null;
  mode?: CheckpointMode;
  saveTopK?: number;
  saveLast?: boolean;
  everyNEpochs?: number;
};
type CheckpointModel = Omit<LightningModuleLike, '_currentOptimizers'> & {
  stateDict(): unknown;
  _currentOptimizers?: CheckpointOptimizer[];
};
type CheckpointOptimizer = {
  stateDict(): unknown;
  loadStateDict(state: unknown): void;
};
type CheckpointPayload = {
  epoch: number;
  globalStep: number;
  modelState: unknown;
  optimizerStates?: unknown[];
};

export class ModelCheckpoint extends Callback {
  private _dirpath: string;
  private _filename: string;
  private _monitor: string | null;
  private _saveTopK: number;
  private _saveLast: boolean;
  private _everyNEpochs: number;
  private _bestK: BestKEntry[];
  private _recent: string[];
  private _compareFn: (a: number, b: number) => number;
  private _bestModelPath: string | null;
  private _lastModelPath: string | null;

  constructor({
    dirpath = './lightning_logs/checkpoints',
    filename = 'epoch={epoch}-step={step}',
    monitor = null,
    mode = 'min',
    saveTopK = 1,
    saveLast = true,
    everyNEpochs = 1,
  }: ModelCheckpointOptions = {}) {
    super();
    this._dirpath = dirpath;
    this._filename = filename;
    this._monitor = monitor;
    this._saveTopK = saveTopK;
    this._saveLast = saveLast;
    this._everyNEpochs = everyNEpochs;
    this._bestK = [];
    this._recent = [];
    this._compareFn = mode === 'min'
      ? (a, b) => a - b
      : (a, b) => b - a;
    this._bestModelPath = null;
    this._lastModelPath = null;
  }

  get bestModelPath(): string | null { return this._bestModelPath; }
  get lastModelPath(): string | null { return this._lastModelPath; }
  get bestKModels(): BestKEntry[] { return this._bestK; }

  onTrainEpochEnd(trainer: TrainerLike, model: CheckpointModel): void {
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
      if (this._saveTopK >= 0) {
        this._recent.push(path);
        while (this._recent.length > this._saveTopK) this._tryDelete(this._recent.shift()!);
      }
      return;
    }

    const metrics = state.epochMetrics!.computeAll();
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
        this._tryDelete(removed!.path);
      }
      this._updateBest();
    }
  }

  private _findInsertIndex(score: number): number {
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

  private _updateBest(): void {
    if (this._bestK.length > 0) {
      this._bestModelPath = this._bestK[0].path;
    }
  }

  private _saveCheckpoint(model: CheckpointModel, trainer: TrainerLike, path: string): void {
    const checkpoint: CheckpointPayload = {
      epoch: trainer.state.epoch,
      globalStep: trainer.state.globalStep,
      modelState: model.stateDict(),
    };

    const optimizers = model._currentOptimizers;
    if (optimizers && optimizers.length > 0) {
      checkpoint.optimizerStates = optimizers.map((opt) => opt.stateDict());
    }

    trainer.callbackConnector!.dispatch('onSaveCheckpoint', trainer, model, checkpoint);

    const tmp = path + '.tmp';
    fs.writeBinary(tmp, serializeCheckpoint(checkpoint));
    fs.rename(tmp, path);
  }

  private _fillTemplate(state: TrainerLike['state']): string {
    return this._filename
      .replace('{epoch}', String(state.epoch))
      .replace('{step}', String(state.globalStep));
  }

  private _ensureDir(): void {
    if (!fs.exists(this._dirpath)) {
      fs.mkdir(this._dirpath);
    }
  }

  private _tryDelete(path: string): void {
    try { fs.remove(path); } catch {}
  }
}

export function loadCheckpoint(path: string): unknown {
  return deserializeCheckpoint(fs.readBinary(path));
}

export function applyCheckpoint(checkpoint: unknown, model: LightningModuleLike, optimizers: CheckpointOptimizer[] = []): unknown {
  const payload = checkpoint as Partial<CheckpointPayload>;
  if (payload.modelState) model.loadStateDict!(payload.modelState);
  if (payload.optimizerStates) {
    const n = Math.min(optimizers.length, payload.optimizerStates.length);
    for (let i = 0; i < n; i++) optimizers[i].loadStateDict(payload.optimizerStates[i]);
  }
  return checkpoint;
}
