import { PassResult } from '../passes/pass.js';

export type TraceEvent = Record<string, unknown> & { level: number; type: string; timestamp?: number };
export type TraceSink = (event: TraceEvent) => void;
export type IRSnapshotFlags = { afterGraphPasses: boolean; afterLowering: boolean; afterScheduling: boolean };
export type TraceLogConfig = Readonly<{ level?: number; sink?: TraceSink; irSnapshot?: Partial<IRSnapshotFlags> }>;

export const TraceLevel = Object.freeze({
  SILENT: 0,
  INFO: 1,
  VERBOSE: 2,
  DEBUG: 3,
});

export class CompilationError {
  phase: string;
  funcName: string;
  message: string;
  passName: string | null;

  constructor(phase: string, funcName: string, message: string, passName?: string | null) {
    this.phase = phase;
    this.funcName = funcName;
    this.message = message;
    this.passName = passName || null;
  }

  toString(): string {
    let s = '[' + this.phase + ']';
    if (this.funcName) s += ' ' + this.funcName;
    if (this.passName) s += ' (' + this.passName + ')';
    s += ': ' + this.message;
    return s;
  }
}

const NOOP: TraceSink = () => {};

export class TraceLog {
  level: number;
  sink: TraceSink;
  irSnapshot: IRSnapshotFlags;

  constructor(config: TraceLogConfig = {}) {
    this.level = config.level ?? TraceLevel.SILENT;
    this.sink = typeof config.sink === 'function' ? config.sink : NOOP;
    this.irSnapshot = {
      afterGraphPasses: false,
      afterLowering: false,
      afterScheduling: false,
      ...(config.irSnapshot || {}),
    };
  }

  emit(event: TraceEvent): void {
    if (event.level > this.level) return;
    event.timestamp = performance.now();
    this.sink(event);
  }

  phaseStart(phase: string): void {
    this.emit({ type: 'phase', action: 'start', phase, level: TraceLevel.INFO });
  }

  phaseEnd(phase: string, durationMs: number): void {
    this.emit({ type: 'phase', action: 'end', phase, durationMs, level: TraceLevel.INFO });
  }

  passRun(passName: string, result: unknown, durationMs: number, opCountBefore: number, opCountAfter: number): void {
    this.emit({
      type: 'pass',
      passName,
      changed: result === PassResult.CHANGED,
      durationMs,
      opCountBefore,
      opCountAfter,
      level: TraceLevel.VERBOSE,
    });
  }

  functionEvent(phase: string, funcName: string, data: Record<string, unknown>): void {
    this.emit({
      type: 'function',
      phase,
      funcName,
      ...data,
      level: TraceLevel.INFO,
    });
  }

  irDump(label: string, text: string): void {
    this.emit({ type: 'ir_snapshot', label, text, level: TraceLevel.DEBUG });
  }

  memoryStats(funcName: string, stats: Record<string, unknown>): void {
    this.emit({ type: 'memory', funcName, ...stats, level: TraceLevel.VERBOSE });
  }

  autotuneStats(funcName: string, stats: Record<string, unknown>): void {
    this.emit({ type: 'autotune', funcName, ...stats, level: TraceLevel.VERBOSE });
  }

  codegenStats(funcName: string, stats: Record<string, unknown>): void {
    this.emit({ type: 'codegen', funcName, ...stats, level: TraceLevel.VERBOSE });
  }

  errorEvent(phase: string, funcName: string, message: string, passName?: string | null): void {
    this.emit({ type: 'error', phase, funcName, message, passName: passName || null, level: TraceLevel.INFO });
  }

  warn(phase: string, funcName: string, message: string, detail?: unknown): void {
    this.emit({ type: 'warning', phase, funcName, message, detail: detail || null, level: TraceLevel.INFO });
  }

  explain(category: string, subject: string, decision: string, reason: string | null, data?: Record<string, unknown>): void {
    this.emit({ type: 'explain', category, subject, decision, reason, ...(data || {}), level: TraceLevel.DEBUG });
  }

  get explainsEnabled(): boolean { return this.level >= TraceLevel.DEBUG; }

  shouldSnapshot(point: keyof IRSnapshotFlags): boolean {
    return this.level >= TraceLevel.DEBUG && !!this.irSnapshot[point];
  }
}

