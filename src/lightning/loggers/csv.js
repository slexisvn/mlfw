import { writeFileSync, appendFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { Logger } from './logger.js';

export class CSVLogger extends Logger {
  constructor({
    saveDir = './lightning_logs',
    name = 'default',
    version = null,
    flushInterval = 10,
  } = {}) {
    super({ name, version: version || 0 });
    this._saveDir = saveDir;
    this._flushInterval = flushInterval;
    this._columns = [];
    this._columnSet = new Set();
    this._buffer = [];
    this._filePath = null;
    this._headerWritten = false;
    this._version = version;
  }

  get logDir() {
    const ver = this._version !== null ? this._version : this._resolveVersion();
    return join(this._saveDir, this._name, `version_${ver}`);
  }

  logMetrics(metrics, step) {
    const row = { step };
    const keys = Object.keys(metrics);
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      row[k] = metrics[k];
      if (!this._columnSet.has(k)) {
        this._columnSet.add(k);
        this._columns.push(k);
        this._headerWritten = false;
      }
    }
    this._buffer.push(row);
    if (this._buffer.length >= this._flushInterval) {
      this._flush();
    }
  }

  logHyperparams(params) {
    this._ensureDir();
    const path = join(this.logDir, 'hparams.json');
    writeFileSync(path, JSON.stringify(params, null, 2));
  }

  finalize() {
    if (this._buffer.length > 0) this._flush();
  }

  _flush() {
    this._ensureDir();
    const path = this._getFilePath();

    if (!this._headerWritten) {
      const header = ['step', ...this._columns].join(',');
      writeFileSync(path, header + '\n');
      this._headerWritten = true;
    }

    const allCols = ['step', ...this._columns];
    const lines = [];
    for (let i = 0; i < this._buffer.length; i++) {
      const row = this._buffer[i];
      const cells = [];
      for (let j = 0; j < allCols.length; j++) {
        const val = row[allCols[j]];
        cells.push(val !== undefined ? String(val) : '');
      }
      lines.push(cells.join(','));
    }
    appendFileSync(path, lines.join('\n') + '\n');
    this._buffer.length = 0;
  }

  _getFilePath() {
    if (!this._filePath) {
      this._filePath = join(this.logDir, 'metrics.csv');
    }
    return this._filePath;
  }

  _ensureDir() {
    const dir = this.logDir;
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
  }

  _resolveVersion() {
    if (this._version !== null) return this._version;
    const baseDir = join(this._saveDir, this._name);
    if (!existsSync(baseDir)) {
      this._version = 0;
      return 0;
    }
    let maxVer = -1;
    try {
      const entries = readdirSync(baseDir);
      for (let i = 0; i < entries.length; i++) {
        const match = entries[i].match(/^version_(\d+)$/);
        if (match) {
          const v = parseInt(match[1], 10);
          if (v > maxVer) maxVer = v;
        }
      }
    } catch { /* empty */ }
    this._version = maxVer + 1;
    return this._version;
  }
}
