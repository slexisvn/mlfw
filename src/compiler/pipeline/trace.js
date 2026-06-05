import { performance } from 'node:perf_hooks';
import { PassResult } from '../passes/pass.js';

export const TraceLevel = Object.freeze({
  SILENT: 0,
  INFO: 1,
  VERBOSE: 2,
  DEBUG: 3,
});

export class CompilationError {
  constructor(phase, funcName, message, passName) {
    this.phase = phase;
    this.funcName = funcName;
    this.message = message;
    this.passName = passName || null;
  }

  toString() {
    let s = '[' + this.phase + ']';
    if (this.funcName) s += ' ' + this.funcName;
    if (this.passName) s += ' (' + this.passName + ')';
    s += ': ' + this.message;
    return s;
  }
}

const NOOP = () => {};

export class TraceLog {
  constructor(config = {}) {
    this.level = config.level ?? TraceLevel.SILENT;
    this.sink = typeof config.sink === 'function' ? config.sink : NOOP;
    this.irSnapshot = {
      afterGraphPasses: false,
      afterLowering: false,
      afterScheduling: false,
      ...(config.irSnapshot || {}),
    };
    this._compileStart = 0;
  }

  emit(event) {
    if (event.level > this.level) return;
    event.timestamp = performance.now();
    this.sink(event);
  }

  phaseStart(phase) {
    this.emit({ type: 'phase', action: 'start', phase, level: TraceLevel.INFO });
  }

  phaseEnd(phase, durationMs) {
    this.emit({ type: 'phase', action: 'end', phase, durationMs, level: TraceLevel.INFO });
  }

  passRun(passName, result, durationMs, opCountBefore, opCountAfter) {
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

  functionEvent(phase, funcName, data) {
    this.emit({
      type: 'function',
      phase,
      funcName,
      ...data,
      level: TraceLevel.INFO,
    });
  }

  irDump(label, text) {
    this.emit({ type: 'ir_snapshot', label, text, level: TraceLevel.DEBUG });
  }

  memoryStats(funcName, stats) {
    this.emit({ type: 'memory', funcName, ...stats, level: TraceLevel.VERBOSE });
  }

  autotuneStats(funcName, stats) {
    this.emit({ type: 'autotune', funcName, ...stats, level: TraceLevel.VERBOSE });
  }

  codegenStats(funcName, stats) {
    this.emit({ type: 'codegen', funcName, ...stats, level: TraceLevel.VERBOSE });
  }

  errorEvent(phase, funcName, message, passName) {
    this.emit({ type: 'error', phase, funcName, message, passName: passName || null, level: TraceLevel.INFO });
  }

  shouldSnapshot(point) {
    return this.level >= TraceLevel.DEBUG && !!this.irSnapshot[point];
  }
}

