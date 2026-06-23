import { Callback } from './callback.js';

const PARTIAL_BLOCKS = ['▏', '▎', '▍', '▌', '▋', '▊', '▉'];

export class ProgressCallback extends Callback {
  constructor({ barLength = 24 } = {}) {
    super();
    this._barLength = barLength;
    this._trainBatchCount = 0;
    this._valBatchCount = 0;
    this._epochStartTime = 0;
    this._lastLen = 0;
    this._active = false;
  }

  onTrainEpochStart(trainer, _model) {
    this._trainBatchCount = 0;
    this._epochStartTime = Date.now();
    const total = this._trainTotal(trainer);
    if (total) this._render('Epoch', trainer.state.epoch + 1, trainer.state.maxEpochs, 0, total, trainer.state);
  }

  onTrainBatchEnd(trainer, _model, _outputs, _batch, _batchIdx) {
    this._trainBatchCount++;
    const total = this._trainTotal(trainer);
    if (total) this._render('Epoch', trainer.state.epoch + 1, trainer.state.maxEpochs, this._trainBatchCount, total, trainer.state);
  }

  onTrainEpochEnd(trainer, _model) {
    const total = this._trainTotal(trainer);
    if (total) this._render('Epoch', trainer.state.epoch + 1, trainer.state.maxEpochs, total, total, trainer.state);
  }

  onTrainEnd(_trainer, _model) {
    if (this._active) process.stdout.write('\n');
    this._active = false;
    this._lastLen = 0;
  }

  onValidationEpochStart(_trainer, _model) {
    this._valBatchCount = 0;
    this._epochStartTime = Date.now();
  }

  onValidationBatchEnd(trainer, _model, _outputs, _batch, _batchIdx) {
    this._valBatchCount++;
    const total = this._valTotal(trainer);
    if (total) this._render('Validation', null, null, this._valBatchCount, total, trainer.state);
  }

  onValidationEnd(trainer, _model) {
    const total = this._valTotal(trainer);
    if (total) this._render('Validation', null, null, total, total, trainer.state);
  }

  _trainTotal(trainer) {
    return trainer.state.numTrainingBatches ?? resolveTotal(trainer.limitTrainBatches);
  }

  _valTotal(trainer) {
    return trainer.state.numValBatches ?? resolveTotal(trainer.limitValBatches);
  }

  _render(label, epoch, maxEpochs, current, total, state) {
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
    process.stdout.write('\r' + line + ' '.repeat(pad));
    this._lastLen = line.length;
    this._active = true;
  }

  _bar(frac) {
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

  _formatProgBarMetrics(state) {
    const pbm = state._progBarMetrics;
    if (!pbm || pbm.size === 0) return '';
    const parts = [];
    for (const [name, value] of pbm) parts.push(`${name}=${formatNum(value)}`);
    return ', ' + parts.join(', ');
  }
}

function resolveTotal(limitConfig) {
  if (limitConfig === null || limitConfig === undefined) return null;
  if (typeof limitConfig === 'number' && limitConfig > 1) return limitConfig;
  return null;
}

function fmtTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatNum(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(4);
}
