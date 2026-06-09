import { Callback } from './callback.js';

export class ProgressCallback extends Callback {
  constructor({ barLength = 30 } = {}) {
    super();
    this._barLength = barLength;
    this._trainBatchCount = 0;
    this._valBatchCount = 0;
    this._epochStartTime = 0;
  }

  onTrainEpochStart(trainer, _model) {
    this._trainBatchCount = 0;
    this._epochStartTime = Date.now();
    const state = trainer.state;
    process.stdout.write(`\nEpoch ${state.epoch + 1}/${state.maxEpochs}\n`);
  }

  onTrainBatchEnd(trainer, _model, _outputs, _batch, _batchIdx) {
    this._trainBatchCount++;
    const total = resolveTotal(trainer.limitTrainBatches);
    if (total !== null) {
      this._renderBar('Train', this._trainBatchCount, total, trainer.state);
    }
  }

  onTrainEpochEnd(trainer, _model) {
    const elapsed = ((Date.now() - this._epochStartTime) / 1000).toFixed(1);
    const metrics = this._formatProgBarMetrics(trainer.state);
    process.stdout.write(`\n  ${elapsed}s${metrics}\n`);
  }

  onValidationEpochStart(_trainer, _model) {
    this._valBatchCount = 0;
  }

  onValidationBatchEnd(trainer, _model, _outputs, _batch, _batchIdx) {
    this._valBatchCount++;
    const total = resolveTotal(trainer.limitValBatches);
    if (total !== null) {
      this._renderBar('Val', this._valBatchCount, total, trainer.state);
    }
  }

  onValidationEnd(trainer, _model) {
    const metrics = this._formatProgBarMetrics(trainer.state);
    if (metrics) process.stdout.write(`  Val${metrics}\n`);
  }

  _renderBar(prefix, current, total, _state) {
    if (!total || total <= 0) return;
    const frac = Math.min(current / total, 1);
    const filled = Math.round(frac * this._barLength);
    const empty = this._barLength - filled;
    const bar = '='.repeat(filled) + (filled < this._barLength ? '>' : '') + ' '.repeat(Math.max(0, empty - 1));
    const pct = Math.round(frac * 100);
    process.stdout.write(`\r  ${prefix} [${bar}] ${pct}% ${current}/${total}`);
  }

  _formatProgBarMetrics(state) {
    const pbm = state._progBarMetrics;
    if (!pbm || pbm.size === 0) return '';
    const parts = [];
    for (const [name, value] of pbm) {
      parts.push(`${name}: ${formatNum(value)}`);
    }
    return ' - ' + parts.join(' - ');
  }
}

function resolveTotal(limitConfig) {
  if (limitConfig === null || limitConfig === undefined) return null;
  if (typeof limitConfig === 'number' && limitConfig > 1) return limitConfig;
  return null;
}

function formatNum(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(4);
}
