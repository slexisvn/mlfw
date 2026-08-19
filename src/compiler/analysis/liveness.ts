import { UseDefAnalysis } from './use_def.js';
import { TensorType, DYNAMIC } from '../ir/graph/types.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { AnalysisCtor, AnalysisDeps } from './analysis_manager.js';
import type { UseDefResult } from './use_def.js';

export type LiveInterval = { start: number; end: number };
type PressureEvent = { idx: number; delta: number; value: Value };

export class LivenessResult {
  intervals: Map<Value, LiveInterval>;
  opIndex: Map<Operation, number>;
  peakPressure: number;
  peakIndex: number;
  liveAtPeak: Set<Value>;

  constructor(intervals: Map<Value, LiveInterval>, opIndex: Map<Operation, number>, peakPressure: number, peakIndex: number, liveAtPeak: Set<Value>) {
    this.intervals = intervals;
    this.opIndex = opIndex;
    this.peakPressure = peakPressure;
    this.peakIndex = peakIndex;
    this.liveAtPeak = liveAtPeak;
  }

  interfere(a: Value, b: Value): boolean {
    if (a === b) return true;
    const intA = this.intervals.get(a);
    const intB = this.intervals.get(b);
    if (intA && intB) {
      return intA.start <= intB.end && intB.start <= intA.end;
    }
    return false;
  }

  intervalOf(value: Value): LiveInterval | null {
    return this.intervals.get(value) || null;
  }
}

function valueBytes(value: Value): number {
  if (!(value.type instanceof TensorType)) return 0;
  const bytes = value.type.sizeInBytes() as number;
  return bytes === DYNAMIC || bytes <= 0 ? 0 : bytes;
}

export class LivenessAnalysis {
  static get name(): string { return 'liveness'; }
  static get depKey(): string { return 'liveness'; }
  static get dependencies(): readonly AnalysisCtor[] { return [UseDefAnalysis as unknown as AnalysisCtor]; }

  static buildIntervals(func: GraphFunction, topo: readonly Operation[]): { intervals: Map<Value, LiveInterval>; opIndex: Map<Operation, number> } {
    const opIndex = new Map<Operation, number>();
    for (let i = 0; i < topo.length; i++) {
      opIndex.set(topo[i], i);
    }

    const intervals = new Map<Value, LiveInterval>();

    for (const arg of func.args) {
      intervals.set(arg, { start: -1, end: -1 });
    }

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      for (let j = 0; j < op.numResults; j++) {
        intervals.set(op.getResult(j), { start: i, end: i });
      }
    }

    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      for (let j = 0; j < op.numOperands; j++) {
        const val = op.getOperand(j);
        const intv = intervals.get(val);
        if (intv && intv.end < i) intv.end = i;
      }
    }

    return { intervals, opIndex };
  }

  static sweepPressure(intervals: ReadonlyMap<Value, LiveInterval>, length: number): { peakPressure: number; peakIndex: number; liveAtPeak: Set<Value> } {
    const events: PressureEvent[] = [];
    for (const [value, intv] of intervals) {
      const bytes = valueBytes(value);
      if (bytes === 0) continue;
      events.push({ idx: intv.start, delta: bytes, value });
      events.push({ idx: intv.end + 1, delta: -bytes, value });
    }
    events.sort((a, b) => a.idx - b.idx || a.delta - b.delta);

    let pressure = 0;
    let peakPressure = 0;
    let peakIndex = 0;
    const live = new Set<Value>();
    const liveAtPeak = new Set<Value>();

    let ei = 0;
    for (let i = -1; i <= length; i++) {
      while (ei < events.length && events[ei].idx <= i) {
        const event = events[ei];
        pressure += event.delta;
        if (event.delta > 0) live.add(event.value);
        else live.delete(event.value);
        ei++;
      }
      if (pressure > peakPressure) {
        peakPressure = pressure;
        peakIndex = i;
        liveAtPeak.clear();
        for (const v of live) liveAtPeak.add(v);
      }
    }

    return { peakPressure, peakIndex, liveAtPeak };
  }

  static compute(func: GraphFunction, deps: AnalysisDeps = {}): LivenessResult {
    const useDef = (deps.useDef as UseDefResult | undefined) || UseDefAnalysis.compute(func);
    const topo = useDef.topologicalOrder;
    const { intervals, opIndex } = LivenessAnalysis.buildIntervals(func, topo);
    const { peakPressure, peakIndex, liveAtPeak } = LivenessAnalysis.sweepPressure(intervals, topo.length);
    return new LivenessResult(intervals, opIndex, peakPressure, peakIndex, liveAtPeak);
  }
}
