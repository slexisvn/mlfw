import { fs } from '#io/fs';
import { joinPath } from '../../runtime/path.js';
import { Logger } from './logger.js';
import type { HyperparameterRecord, LoggerOptions, NumericMetricRecord } from '../types.js';

type CSVLoggerOptions = LoggerOptions & {
  saveDir?: string;
  flushInterval?: number;
};

export class CSVLogger extends Logger {
  private _saveDir: string;
  private _flushInterval: number;
  private _columns: string[];
  private _columnSet: Set<string>;
  private _buffer: NumericMetricRecord[];
  private _filePath: string | null;
  private _headerWritten: boolean;
  protected _version: number | null;

  constructor({
    saveDir = './lightning_logs',
    name = 'default',
    version = null,
    flushInterval = 10,
  }: CSVLoggerOptions = {}) {
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

  get logDir(): string {
    const ver = this._version !== null ? this._version : this._resolveVersion();
    return joinPath(this._saveDir, this._name, `version_${ver}`);
  }

  logMetrics(metrics: NumericMetricRecord, step: number): void {
    const row: NumericMetricRecord = { step };
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

  logHyperparams(params: HyperparameterRecord): void {
    this._ensureDir();
    const path = joinPath(this.logDir, 'hparams.json');
    fs.writeFile(path, JSON.stringify(params, null, 2));
  }

  finalize(): void {
    if (this._buffer.length > 0) this._flush();
  }

  private _flush(): void {
    this._ensureDir();
    const path = this._getFilePath();

    if (!this._headerWritten) {
      const header = ['step', ...this._columns].join(',');
      fs.writeFile(path, header + '\n');
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
    fs.appendFile(path, lines.join('\n') + '\n');
    this._buffer.length = 0;
  }

  private _getFilePath(): string {
    if (!this._filePath) {
      this._filePath = joinPath(this.logDir, 'metrics.csv');
    }
    return this._filePath;
  }

  private _ensureDir(): void {
    const dir = this.logDir;
    if (!fs.exists(dir)) {
      fs.mkdir(dir);
    }
  }

  private _resolveVersion(): number {
    if (this._version !== null) return this._version;
    const baseDir = joinPath(this._saveDir, this._name);
    if (!fs.exists(baseDir)) {
      this._version = 0;
      return 0;
    }
    let maxVer = -1;
    try {
      const entries = fs.readdir(baseDir);
      for (let i = 0; i < entries.length; i++) {
        const match = entries[i].match(/^version_(\d+)$/);
        if (match) {
          const v = parseInt(match[1], 10);
          if (v > maxVer) maxVer = v;
        }
      }
    } catch {}
    this._version = maxVer + 1;
    return this._version;
  }
}
