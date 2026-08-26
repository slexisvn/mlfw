import { Schedule } from 'mlfw/compiler/schedule/schedule.js';
import { ScheduleTrace } from 'mlfw/compiler/schedule/trace.js';
import { clonePrimFunc } from 'mlfw/compiler/autotune/tune_ir.js';
import { takeSnapshot } from './snapshot.js';
import { PRIMITIVE_NOTES } from '../catalog/primitives.js';
import type { PrimFunc } from 'mlfw/compiler/ir/tensor/nodes.js';
import type { SerializedStep } from 'mlfw/compiler/schedule/trace.js';
import type { CompileStep, Snapshot, TraceEventLite } from '../protocol.js';

const ARG_LIMIT = 3;

export function captureBaselines(target: unknown): Map<string, PrimFunc> {
  const baselines = new Map<string, PrimFunc>();
  if (!target || typeof (target as Iterable<PrimFunc>)[Symbol.iterator] !== 'function') return baselines;
  for (const func of target as Iterable<PrimFunc>) baselines.set(func.name, clonePrimFunc(func));
  return baselines;
}

function formatArg(arg: unknown): string {
  if (Array.isArray(arg)) return `[${arg.map(formatArg).join(', ')}]`;
  if (typeof arg === 'string') return arg;
  return String(arg);
}

function label(step: SerializedStep): string {
  const args = step.args.slice(0, ARG_LIMIT).map(formatArg);
  if (step.args.length > ARG_LIMIT) args.push('…');
  return `${step.primitive}(${args.join(', ')})`;
}

function note(primitive: string): TraceEventLite {
  const entry = PRIMITIVE_NOTES[primitive];
  return {
    type: 'explain',
    category: 'schedule',
    subject: primitive,
    decision: entry ? entry.decision : 'rewrote the loop nest',
    reason: entry ? entry.reason : null,
  };
}

export function primitiveSteps(
  baselines: ReadonlyMap<string, PrimFunc>,
  events: readonly TraceEventLite[],
  phase: string,
): CompileStep[] {
  const steps: CompileStep[] = [];

  for (const event of events) {
    if (event.type !== 'schedule_trace') continue;
    const funcName = String(event.funcName);
    const baseline = baselines.get(funcName);
    if (!baseline) continue;

    const work = clonePrimFunc(baseline);
    const schedule = new Schedule(work);
    const recorded = ScheduleTrace.deserialize(event.steps as SerializedStep[]);
    let before: Snapshot = takeSnapshot([work], 'tir');

    try {
      recorded.replayEach(schedule as never, (step) => {
        const after = takeSnapshot([schedule.func], 'tir');
        steps.push({
          index: 0,
          kind: 'primitive',
          parent: funcName,
          unit: null,
          level: 'tir',
          phase,
          pass: label(step.serialize()),
          outcome: before.text === after.text ? 'unchanged' : 'changed',
          durationMs: 0,
          before,
          after,
          events: [note(step.primitive)],
        });
        before = after;
      });
    } catch (error) {
      steps.push({
        index: 0,
        kind: 'primitive',
        parent: funcName,
        unit: null,
        level: 'tir',
        phase,
        pass: `replay stopped after ${steps.length} primitive${steps.length === 1 ? '' : 's'}`,
        outcome: 'failed',
        durationMs: 0,
        before,
        after: before,
        events: [{
          type: 'warning',
          phase,
          message: error instanceof Error ? error.message : String(error),
        }],
      });
    }
  }

  return steps;
}
