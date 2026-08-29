import { PassResult } from 'mlfw/compiler/passes/pass.js';
import { cloneTensorIR } from 'mlfw/compiler/autotune/tune_ir.js';
import { takeSnapshot } from './snapshot.js';
import { invariantsOf, verifyReport } from './verify.js';
import { sourceSnapshot } from './source_nest.js';
import { captureBaselines, primitiveSteps } from './schedule_steps.js';
import { levelLabel } from '../catalog/naming.js';
import { REPLAYABLE_PASSES } from '../catalog/passes.js';
import type { PrimFunc } from 'mlfw/compiler/ir/tensor/nodes.js';
import type { BufferLifetime, CompileStep, IRLevelName, Kernel, MemoryPlan, PassOutcome, SkippedPass, Snapshot, TraceEventLite, TuningCandidate, TuningParams, TuningRound } from '../protocol.js';

const OUTCOMES: Record<number, PassOutcome> = {
  [PassResult.UNCHANGED]: 'unchanged',
  [PassResult.CHANGED]: 'changed',
  [PassResult.FAILED]: 'failed',
};

const GRAPH_PHASE_FALLBACK = 'graphPasses';
const NESTED_LEVELS: ReadonlySet<string> = new Set(['tir', 'lir']);
const CAPTURE_NODE_CAP = 20000;

export type CapturedIR = { level: IRLevelName; before: NestedFunc[]; after: NestedFunc[] };

type NestedFunc = { name: string; body?: unknown };

function captureFuncs(target: unknown): NestedFunc[] {
  if (!target || typeof (target as Iterable<unknown>)[Symbol.iterator] !== 'function') return [];
  const funcs: NestedFunc[] = [];
  for (const func of target as Iterable<NestedFunc>) funcs.push(cloneTensorIR(func as never) as unknown as NestedFunc);
  return funcs;
}

type OpenStep = {
  pass: string;
  level: IRLevelName;
  phase: string;
  startedAt: number;
  eventMark: number;
  before: Snapshot;
  invariants: string[] | null;
  captured: NestedFunc[] | null;
  unit: string | null;
  baselines: Map<string, PrimFunc> | null;
};

export class CompileRecorder {
  readonly steps: CompileStep[] = [];
  readonly events: TraceEventLite[] = [];
  readonly bodies = new Map<number, CapturedIR>();

  private readonly captures = new Map<number, CapturedIR>();

  private readonly phases: string[] = [];
  private readonly open: OpenStep[] = [];
  private readonly verifying: boolean;
  private onTracingDone: (() => void) | null;

  constructor(verifying: boolean, onTracingDone: () => void = () => {}) {
    this.verifying = verifying;
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
      invariants: this.verifying ? invariantsOf(level, target) : null,
      captured: NESTED_LEVELS.has(level) ? captureFuncs(target) : null,
      unit: unitOf(target),
      baselines: REPLAYABLE_PASSES.has(pass.name) ? captureBaselines(target) : null,
    });
  };

  runAfterPass = (pass: { name: string }, target: unknown, level: IRLevelName, result: number | null): void => {
    const open = this.open.pop();
    if (!open) return;
    const after = takeSnapshot(target, level);
    const events = this.events.slice(open.eventMark);

    if (open.captured && open.before.ops <= CAPTURE_NODE_CAP && after.ops <= CAPTURE_NODE_CAP) {
      this.captures.set(this.steps.length, { level, before: open.captured, after: captureFuncs(target) });
    }

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
      verify: open.invariants === null ? null : verifyReport(open.invariants, invariantsOf(level, target)),
      interpretable: false,
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
      verify: null,
      interpretable: false,
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
          verify: null,
          interpretable: false,
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
          verify: null,
          interpretable: false,
        });
      }
      const captured = this.captures.get(step.index);
      if (captured) this.bodies.set(timeline.length, captured);
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
        verify: null,
        interpretable: false,
      });
    }

    return timeline.map((step, index) => ({ ...step, index, interpretable: this.bodies.has(index) }));
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

  skippedPasses(): SkippedPass[] {
    const skipped: SkippedPass[] = [];
    const seen = new Set<string>();

    for (const event of this.events) {
      if (event.type !== 'pass_skipped') continue;
      const pass = String(event.passName);
      if (seen.has(pass)) continue;
      seen.add(pass);
      skipped.push({ pass, level: event.irLevel as IRLevelName });
    }

    return skipped;
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
        verify: open.invariants === null ? null : verifyReport(open.invariants, open.invariants),
        interpretable: false,
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
