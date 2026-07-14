import { Callback } from './callback.js';
import type { TrainerLike } from '../types.js';

const PARTIAL_BLOCKS = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];

type ProgressOptions = {
  barLength?: number;
};

function stdoutWrite(text: string): void {
  (globalThis as unknown as { process: { stdout: { write(value: string): void } } }).process.stdout.write(text);
}

export class ProgressCallback extends Callback {
  private _barLength: number;
  private _trainBatchCount: number;
  private _valBatchCount: number;
  private _epochStartTime: number;
  private _lastLen: number;
  private _active: boolean;

  constructor({ barLength = 24 }: ProgressOptions = {}) {
    super();
    this._barLength = barLength;
    this._trainBatchCount = 0;
    this._valBatchCount = 0;
    this._epochStartTime = 0;
    this._lastLen = 0;
    this._active = false;
  }

  onTrainEpochStart(trainer: TrainerLike, _model: unknown): void {
    this._trainBatchCount = 0;
    this._epochStartTime = Date.now();
    const total = this._trainTotal(trainer);
    if (total) this._render('Epoch', trainer.state.epoch + 1, trainer.state.maxEpochs, 0, total, trainer.state);
  }

  onTrainBatchEnd(trainer: TrainerLike, _model: unknown, _outputs: unknown, _batch: unknown, _batchIdx: unknown): void {
    this._trainBatchCount++;
    const total = this._trainTotal(trainer);
    if (total) this._render('Epoch', trainer.state.epoch + 1, trainer.state.maxEpochs, this._trainBatchCount, total, trainer.state);
  }

  onTrainEpochEnd(trainer: TrainerLike, _model: unknown): void {
    const total = this._trainTotal(trainer);
    if (total) this._render('Epoch', trainer.state.epoch + 1, trainer.state.maxEpochs, total, total, trainer.state);
  }

  onTrainEnd(_trainer: unknown, _model: unknown): void {
    if (this._active) stdoutWrite('\n');
    this._active = false;
    this._lastLen = 0;
  }

  onValidationEpochStart(_trainer: unknown, _model: unknown): void {
    this._valBatchCount = 0;
    this._epochStartTime = Date.now();
  }

  onValidationBatchEnd(trainer: TrainerLike, _model: unknown, _outputs: unknown, _batch: unknown, _batchIdx: unknown): void {
    this._valBatchCount++;
    const total = this._valTotal(trainer);
    if (total) this._render('Validation', null, null, this._valBatchCount, total, trainer.state);
  }

  onValidationEnd(trainer: TrainerLike, _model: unknown): void {
    const total = this._valTotal(trainer);
    if (total) this._render('Validation', null, null, total, total, trainer.state);
  }

  private _trainTotal(trainer: TrainerLike): number | null {
    return trainer.state.numTrainingBatches ?? resolveTotal(trainer.limitTrainBatches);
  }

  private _valTotal(trainer: TrainerLike): number | null {
    return trainer.state.numValBatches ?? resolveTotal(trainer.limitValBatches);
  }

  private _render(label: string, epoch: number | null, maxEpochs: number | null | undefined, current: number, total: number, state: TrainerLike['state']): void {
    const frac = total > 0 ? Math.min(current / total, 1) : 0;
    const pct = String(Math.round(frac * 100)).padStart(3, ' ');
    const bar = this._bar(frac);
    const head = epoch !== null ? `${label} ${epoch}/${maxEpochs}` : label;
    const elapsed = (Date.now() - this._epochStartTime) / 1000;
    const rate = elapsed > 0 ? current / elapsed : 0;
    const eta = rate > 0 ? (total - current) / rate : 0;
    const timing = `${fmtTime(elapsed)}<${fmtTime(eta)}, ${rate.toFixed(2)}it/s`;
    const metrics = this._formatProgBarMetrics(state);
    const line = `${head}: ${pct}%|${bar}| ${current}/${total} [${timing}${metrics}]`;
    const pad = Math.max(0, this._lastLen - line.length);
    stdoutWrite('\r' + line + ' '.repeat(pad));
    this._lastLen = line.length;
    this._active = true;
  }

  private _bar(frac: number): string {
    const width = this._barLength;
    const exact = frac * width;
    let whole = Math.floor(exact);
    let eighths = Math.round((exact - whole) * 8);
    if (eighths === 8) { whole += 1; eighths = 0; }
    if (whole >= width) return '█'.repeat(width);
    const partial = eighths > 0 ? PARTIAL_BLOCKS[eighths - 1] : '';
    const pad = width - whole - (partial ? 1 : 0);
    return '█'.repeat(whole) + partial + ' '.repeat(pad);
  }

  private _formatProgBarMetrics(state: TrainerLike['state']): string {
    const pbm = state._progBarMetrics;
    if (!pbm || pbm.size === 0) return '';
    const parts = [];
    for (const [name, value] of pbm) parts.push(`${name}=${formatNum(value)}`);
    return ', ' + parts.join(', ');
  }
}

function resolveTotal(limitConfig: number | null | undefined): number | null {
  if (limitConfig === null || limitConfig === undefined) return null;
  if (typeof limitConfig === 'number' && limitConfig > 1) return limitConfig;
  return null;
}

function fmtTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatNum(v: unknown): string {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(4);
}
