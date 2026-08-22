import { computeWorkloadKey } from './workload_key.js';
import type { PrimFunc } from '../ir/tensor/nodes.js';
import type { ScheduleTarget } from '../schedule/gpu_matmul_schedule.js';
import type { SketchParams } from './sketch.js';
import type { SerializedStep } from '../schedule/trace.js';

export type SerializedRecord = {
  workloadKey: string;
  sketchName: string;
  params: SketchParams;
  score: number;
  traceData: SerializedStep[] | null;
  version: number;
  timestamp: number;
  medianMs: number | null;
  minMs: number | null;
};

export type SerializedDatabase = { version: number; codegenVersion?: string; scheduleSemanticsVersion?: string; entries: SerializedRecord[] };

export type FsLike = {
  writeFile(path: string, data: string): void;
  readFile(path: string): string;
  exists(path: string): boolean;
};

export const CODEGEN_VERSION = 'mlfw-codegen-1';
export const SCHEDULE_SEMANTICS_VERSION = 'mlfw-schedule-2';

export class TuningRecord {
  workloadKey: string;
  sketchName: string;
  params: SketchParams;
  score: number;
  traceData: SerializedStep[] | null;
  version: number;
  timestamp: number;
  medianMs: number | null;
  minMs: number | null;

  constructor(workloadKey: string, sketchName: string, params: SketchParams, score: number, traceData: SerializedStep[] | null, version: number) {
    this.workloadKey = workloadKey;
    this.sketchName = sketchName;
    this.params = params;
    this.score = score;
    this.traceData = traceData;
    this.version = version;
    this.timestamp = Date.now();
    this.medianMs = null;
    this.minMs = null;
  }
}

function rankRecords(a: TuningRecord, b: TuningRecord): number {
  const am = a.medianMs != null ? 1 : 0;
  const bm = b.medianMs != null ? 1 : 0;
  if (am !== bm) return bm - am;
  if (am === 1) return (a.medianMs as number) - (b.medianMs as number);
  return b.score - a.score;
}

export class TuningDatabase {
  version: number;
  private _records: Map<string, TuningRecord[]>;

  constructor(version = 1) {
    this.version = version;
    this._records = new Map();
  }

  computeWorkloadKey(primFunc: PrimFunc, blockName: string, target: ScheduleTarget, blockMap: never | null = null): string {
    return computeWorkloadKey(primFunc, blockName, target, blockMap);
  }

  store(workloadKey: string, record: TuningRecord): void {
    let list = this._records.get(workloadKey);
    if (!list) {
      list = [];
      this._records.set(workloadKey, list);
    }
    list.push(record);
    list.sort(rankRecords);
    if (list.length > 10) list.length = 10;
  }

  lookup(workloadKey: string): TuningRecord | null {
    const list = this._records.get(workloadKey);
    if (!list || list.length === 0) return null;
    return list[0];
  }

  lookupTopK(workloadKey: string, k = 5): TuningRecord[] {
    const list = this._records.get(workloadKey);
    if (!list) return [];
    return list.slice(0, k);
  }

  has(workloadKey: string): boolean {
    return this._records.has(workloadKey) && (this._records.get(workloadKey) as TuningRecord[]).length > 0;
  }

  get size(): number {
    let count = 0;
    for (const [, list] of this._records) count += list.length;
    return count;
  }

  serialize(): SerializedDatabase {
    const entries: SerializedRecord[] = [];
    for (const records of this._records.values()) {
      for (const r of records) {
        entries.push({
          workloadKey: r.workloadKey,
          sketchName: r.sketchName,
          params: r.params,
          score: r.score,
          traceData: r.traceData,
          version: r.version,
          timestamp: r.timestamp,
          medianMs: r.medianMs,
          minMs: r.minMs
        });
      }
    }
    return { version: this.version, codegenVersion: CODEGEN_VERSION, scheduleSemanticsVersion: SCHEDULE_SEMANTICS_VERSION, entries };
  }

  static deserialize(data: SerializedDatabase): TuningDatabase {
    const db = new TuningDatabase(data.version);
    if (data.codegenVersion !== undefined && data.codegenVersion !== CODEGEN_VERSION) {
      return db;
    }
    if (data.scheduleSemanticsVersion !== SCHEDULE_SEMANTICS_VERSION) {
      return db;
    }
    for (const entry of data.entries) {
      const record = new TuningRecord(
        entry.workloadKey, entry.sketchName, entry.params,
        entry.score, entry.traceData, entry.version
      );
      record.timestamp = entry.timestamp;
      record.medianMs = entry.medianMs ?? null;
      record.minMs = entry.minMs ?? null;
      db.store(entry.workloadKey, record);
    }
    return db;
  }

  saveToFile(path: string, fsImpl: FsLike): string {
    fsImpl.writeFile(path, JSON.stringify(this.serialize()));
    return path;
  }

  static loadFromFile(path: string, fsImpl: FsLike): TuningDatabase {
    if (!fsImpl.exists(path)) return new TuningDatabase();
    return TuningDatabase.deserialize(JSON.parse(fsImpl.readFile(path)));
  }

  clear(): void {
    this._records.clear();
  }
}
