import { computeWorkloadKey } from './workload_key.js';

export class TuningRecord {
  constructor(workloadKey, sketchName, params, score, traceData, version) {
    this.workloadKey = workloadKey;
    this.sketchName = sketchName;
    this.params = params;
    this.score = score;
    this.traceData = traceData;
    this.version = version;
    this.timestamp = Date.now();
  }
}

export class TuningDatabase {
  constructor(version = 1) {
    this.version = version;
    this._records = new Map();
  }

  computeWorkloadKey(primFunc, blockName, target, blockMap = null) {
    return computeWorkloadKey(primFunc, blockName, target, blockMap);
  }

  store(workloadKey, record) {
    let list = this._records.get(workloadKey);
    if (!list) {
      list = [];
      this._records.set(workloadKey, list);
    }
    list.push(record);
    list.sort((a, b) => b.score - a.score);
    if (list.length > 10) list.length = 10;
  }

  lookup(workloadKey) {
    const list = this._records.get(workloadKey);
    if (!list || list.length === 0) return null;
    return list[0];
  }

  lookupTopK(workloadKey, k = 5) {
    const list = this._records.get(workloadKey);
    if (!list) return [];
    return list.slice(0, k);
  }

  has(workloadKey) {
    return this._records.has(workloadKey) && this._records.get(workloadKey).length > 0;
  }

  get size() {
    let count = 0;
    for (const [, list] of this._records) count += list.length;
    return count;
  }

  serialize() {
    const entries = [];
    for (const [key, records] of this._records) {
      for (const r of records) {
        entries.push({
          workloadKey: r.workloadKey,
          sketchName: r.sketchName,
          params: r.params,
          score: r.score,
          traceData: r.traceData,
          version: r.version,
          timestamp: r.timestamp
        });
      }
    }
    return { version: this.version, entries };
  }

  static deserialize(data) {
    const db = new TuningDatabase(data.version);
    for (const entry of data.entries) {
      const record = new TuningRecord(
        entry.workloadKey, entry.sketchName, entry.params,
        entry.score, entry.traceData, entry.version
      );
      record.timestamp = entry.timestamp;
      db.store(entry.workloadKey, record);
    }
    return db;
  }

  clear() {
    this._records.clear();
  }
}
