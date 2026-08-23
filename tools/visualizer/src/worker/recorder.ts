import { PassResult } from 'mlfw/compiler/passes/pass.js';
import { takeSnapshot } from './snapshot.js';
import { sourceSnapshot } from './source_nest.js';
import type { CompileStep, IRLevelName, Kernel, PassOutcome, Snapshot, TraceEventLite } from '../protocol.js';

const OUTCOMES: Record<number, PassOutcome> = {
  [PassResult.UNCHANGED]: 'unchanged',
  [PassResult.CHANGED]: 'changed',
  [PassResult.FAILED]: 'failed',
};

const GRAPH_PHASE_FALLBACK = 'graphPasses';

const LEVEL_LABEL: Record<IRLevelName, string> = {
  'graph-module': 'graph IR',
  'graph-func': 'graph IR',
  tir: 'tensor IR',
  lir: 'low-level IR',
};

type OpenStep = {
  pass: string;
  level: IRLevelName;
  phase: string;
  startedAt: number;
  eventMark: number;
  before: Snapshot;
};

export class CompileRecorder {
  readonly steps: CompileStep[] = [];
  readonly events: TraceEventLite[] = [];

  private readonly phases: string[] = [];
  private readonly open: OpenStep[] = [];

  sink = (event: TraceEventLite): void => {
    this.events.push(event);
    if (event.type !== 'phase') return;
    if (event.action === 'start') this.phases.push(String(event.phase));
    else if (event.action === 'end') this.phases.pop();
  };

  runBeforePass = (pass: { name: string }, target: unknown, level: IRLevelName): void => {
    this.open.push({
      pass: pass.name,
      level,
      phase: this.currentPhase(level),
      startedAt: performance.now(),
      eventMark: this.events.length,
      before: takeSnapshot(target, level),
    });
  };

  runAfterPass = (pass: { name: string }, target: unknown, level: IRLevelName, result: number | null): void => {
    const open = this.open.pop();
    if (!open) return;
    const after = takeSnapshot(target, level);

    this.steps.push({
      index: this.steps.length,
      kind: 'pass',
      level,
      phase: open.phase,
      pass: open.pass,
      outcome: result === null ? diffOutcome(open.before, after) : OUTCOMES[result],
      durationMs: performance.now() - open.startedAt,
      before: open.before,
      after,
      events: this.events.slice(open.eventMark),
    });
  };

  timeline(kernels: readonly Kernel[] = []): CompileStep[] {
    const first = this.steps[0];
    if (!first) return this.steps;

    const timeline: CompileStep[] = [{
      index: 0,
      kind: 'input',
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
      if (previous.level !== step.level) {
        timeline.push({
          index: 0,
          kind: 'lowering',
          level: step.level,
          phase: 'lowering',
          pass: `${LEVEL_LABEL[previous.level]} → ${LEVEL_LABEL[step.level]}`,
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
        level: last.level,
        phase: 'codegen',
        pass: `${LEVEL_LABEL[last.level]} → kernel source`,
        outcome: 'changed',
        durationMs: 0,
        before: last.after,
        after: sourceSnapshot(kernels),
        events: [],
      });
    }

    return timeline.map((step, index) => ({ ...step, index }));
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

function diffOutcome(before: Snapshot, after: Snapshot): PassOutcome {
  return before.text === after.text ? 'unchanged' : 'changed';
}
