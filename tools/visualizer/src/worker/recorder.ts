import { PassResult } from 'mlfw/compiler/passes/pass.js';
import { takeSnapshot } from './snapshot.js';
import { sourceSnapshot } from './source_nest.js';
import { captureBaselines, primitiveSteps } from './schedule_steps.js';
import { levelLabel } from '../catalog/naming.js';
import { REPLAYABLE_PASSES } from '../catalog/passes.js';
import type { PrimFunc } from 'mlfw/compiler/ir/tensor/nodes.js';
import type { BufferLifetime, CompileStep, IRLevelName, Kernel, MemoryPlan, PassOutcome, Snapshot, TraceEventLite, TuningCandidate, TuningParams, TuningRound } from '../protocol.js';

const OUTCOMES: Record<number, PassOutcome> = {
  [PassResult.UNCHANGED]: 'unchanged',
  [PassResult.CHANGED]: 'changed',
  [PassResult.FAILED]: 'failed',
};

const GRAPH_PHASE_FALLBACK = 'graphPasses';

type OpenStep = {
  pass: string;
  level: IRLevelName;
  phase: string;
  startedAt: number;
  eventMark: number;
  before: Snapshot;
  unit: string | null;
  baselines: Map<string, PrimFunc> | null;
};

export class CompileRecorder {
  readonly steps: CompileStep[] = [];
  readonly events: TraceEventLite[] = [];

  private readonly phases: string[] = [];
  private readonly open: OpenStep[] = [];
  private onTracingDone: (() => void) | null;

  constructor(onTracingDone: () => void = () => {}) {
    this.onTracingDone = onTracingDone;
  }

  sink = (event: TraceEventLite): void => {
    this.events.push(event);
    if (event.type !== 'phase') return;
    if (event.action === 'start') this.phases.push(String(event.phase));
    else if (event.action === 'end') this.phases.pop();
  };

  runBeforePass = (pass: { name: string }, target: unknown, level: IRLevelName): void => {
    if (this.onTracingDone) {
      this.onTracingDone();
      this.onTracingDone = null;
    }
    this.open.push({
      pass: pass.name,
      level,
      phase: this.currentPhase(level),
      startedAt: performance.now(),
      eventMark: this.events.length,
      before: takeSnapshot(target, level),
      unit: unitOf(target),
      baselines: REPLAYABLE_PASSES.has(pass.name) ? captureBaselines(target) : null,
    });
  };

  runAfterPass = (pass: { name: string }, target: unknown, level: IRLevelName, result: number | null): void => {
    const open = this.open.pop();
    if (!open) return;
    const after = takeSnapshot(target, level);
    const events = this.events.slice(open.eventMark);

    this.steps.push({
      index: this.steps.length,
      kind: 'pass',
      parent: null,
      unit: open.unit,
      level,
      phase: open.phase,
      pass: open.pass,
      outcome: result === null ? diffOutcome(open.before, after) : OUTCOMES[result],
      durationMs: performance.now() - open.startedAt,
      before: open.before,
      after,
      events,
    });

    if (open.baselines) {
      for (const step of primitiveSteps(open.baselines, events, open.phase)) {
        this.steps.push({ ...step, index: this.steps.length, unit: open.unit });
      }
    }
  };

  timeline(kernels: readonly Kernel[] = []): CompileStep[] {
    const first = this.steps[0];
    if (!first) return this.steps;

    let carried: string | null = null;
    for (const step of this.steps) {
      if (step.unit === null) step.unit = carried;
      else carried = step.unit;
    }

    const timeline: CompileStep[] = [{
      index: 0,
      kind: 'input',
      parent: null,
      unit: first.unit,
      level: first.level,
      phase: 'input',
      pass: 'traced graph',
      outcome: 'unchanged',
      durationMs: 0,
      before: first.before,
      after: first.before,
      events: [],
    }];

    for (const step of this.steps) {
      const previous = timeline[timeline.length - 1];
      if (previous.unit !== step.unit && previous.unit !== null && step.unit !== null) {
        timeline.push({
          index: 0,
          kind: 'lowering',
          parent: null,
          unit: step.unit,
          level: step.level,
          phase: 'handoff',
          pass: `${previous.unit} → ${step.unit}`,
          outcome: 'changed',
          durationMs: 0,
          before: previous.after,
          after: step.before,
          events: [],
        });
      } else if (previous.level !== step.level) {
        timeline.push({
          index: 0,
          kind: 'lowering',
          parent: null,
          unit: step.unit,
          level: step.level,
          phase: 'lowering',
          pass: `${levelLabel(previous.level)} → ${levelLabel(step.level)}`,
          outcome: 'changed',
          durationMs: 0,
          before: previous.after,
          after: step.before,
          events: [],
        });
      }
      timeline.push(step);
    }

    const last = timeline[timeline.length - 1];
    if (kernels.length > 0 && last) {
      timeline.push({
        index: 0,
        kind: 'lowering',
        parent: null,
        unit: last.unit,
        level: last.level,
        phase: 'codegen',
        pass: `${levelLabel(last.level)} → kernel source`,
        outcome: 'changed',
        durationMs: 0,
        before: last.after,
        after: sourceSnapshot(kernels),
        events: [],
      });
    }

    return timeline.map((step, index) => ({ ...step, index }));
  }

  memoryPlans(): MemoryPlan[] {
    const plans: MemoryPlan[] = [];
    for (const event of this.events) {
      if (event.type !== 'memory_plan') continue;
      plans.push({
        func: String(event.funcName),
        peakMemory: Number(event.peakMemory),
        totalBytesIfNeverShared: Number(event.totalBytesIfNeverShared),
        steps: Number(event.steps),
        buffers: event.buffers as BufferLifetime[],
      });
    }
    return plans;
  }

  tuningRounds(): TuningRound[] {
    const rounds: TuningRound[] = [];
    for (const event of this.events) {
      if (event.type !== 'autotune_round') continue;
      rounds.push({
        func: String(event.funcName),
        blockName: String(event.blockName),
        round: Number(event.round),
        measured: event.measured === true,
        scores: event.scores as TuningCandidate[],
        bestSketch: event.bestSketch === null ? null : String(event.bestSketch),
        bestParams: (event.bestParams ?? null) as TuningParams | null,
        bestScore: event.bestScore === null ? null : Number(event.bestScore),
        bestMedianMs: event.bestMedianMs === null || event.bestMedianMs === undefined
          ? null
          : Number(event.bestMedianMs),
      });
    }
    return rounds;
  }

  currentOpenPass(): string | null {
    const open = this.open[this.open.length - 1];
    return open ? open.pass : null;
  }

  closeOpenSteps(): void {
    while (this.open.length > 0) {
      const open = this.open.pop() as OpenStep;
      this.steps.push({
        index: this.steps.length,
        kind: 'pass',
        parent: null,
        unit: open.unit,
        level: open.level,
        phase: open.phase,
        pass: open.pass,
        outcome: 'failed',
        durationMs: performance.now() - open.startedAt,
        before: open.before,
        after: open.before,
        events: this.events.slice(open.eventMark),
      });
    }
  }

  private currentPhase(level: IRLevelName): string {
    const top = this.phases[this.phases.length - 1];
    if (top && top !== 'compile') return top;
    return level === 'graph-module' || level === 'graph-func' ? GRAPH_PHASE_FALLBACK : level;
  }
}

function unitOf(target: unknown): string | null {
  const named = target as { name?: unknown } | null;
  return named && typeof named.name === 'string' ? named.name : null;
}

function diffOutcome(before: Snapshot, after: Snapshot): PassOutcome {
  return before.text === after.text ? 'unchanged' : 'changed';
}
