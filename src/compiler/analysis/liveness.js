import { UseDefAnalysis } from './use_def.js';
import { TensorType, DYNAMIC } from '../ir/graph/types.js';

export class LivenessResult {
  constructor(liveIn, liveOut, intervals, opIndex, peakPressure, peakOp, pressureAtOp) {
    this.liveIn = liveIn;
    this.liveOut = liveOut;
    this.intervals = intervals;
    this.opIndex = opIndex;
    this.peakPressure = peakPressure;
    this.peakOp = peakOp;
    this.pressureAtOp = pressureAtOp;
  }

  interfere(a, b) {
    if (a === b) return true;
    const intA = this.intervals.get(a);
    const intB = this.intervals.get(b);
    if (intA && intB) {
      return intA.start <= intB.end && intB.start <= intA.end;
    }
    return false;
  }

  liveAtOp(op) {
    return this.liveIn.get(op) || new Set();
  }

  intervalOf(value) {
    return this.intervals.get(value) || null;
  }
}

export class LivenessAnalysis {
  static get name() { return 'liveness'; }
  static get depKey() { return 'liveness'; }
  static get dependencies() { return [UseDefAnalysis]; }

  static buildIntervals(func, topo) {
    const opIndex = new Map();
    for (let i = 0; i < topo.length; i++) {
      opIndex.set(topo[i], i);
    }

    const intervals = new Map();

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

  static compute(func, deps = {}) {
    const useDef = deps.useDef || UseDefAnalysis.compute(func);
    const topo = useDef.topologicalOrder;

    const { intervals, opIndex } = LivenessAnalysis.buildIntervals(func, topo);

    const liveIn = new Map();
    const liveOut = new Map();
    for (const op of topo) {
      liveIn.set(op, new Set());
      liveOut.set(op, new Set());
    }

    for (let i = topo.length - 1; i >= 0; i--) {
      const op = topo[i];
      const currentOut = liveOut.get(op);

      const users = useDef.opUsers.get(op);
      if (users) {
        for (const user of users) {
          const userIn = liveIn.get(user);
          if (userIn) {
            for (const v of userIn) {
              currentOut.add(v);
            }
          }
        }
      }

      const currentIn = liveIn.get(op);
      for (const v of currentOut) {
        currentIn.add(v);
      }
      for (let j = 0; j < op.numResults; j++) {
        currentIn.delete(op.getResult(j));
      }
      for (let j = 0; j < op.numOperands; j++) {
        currentIn.add(op.getOperand(j));
      }
    }

    for (const [op, idx] of opIndex) {
      const activeVars = liveIn.get(op);
      for (const v of activeVars) {
        const intv = intervals.get(v);
        if (intv && intv.end < idx) intv.end = idx;
      }
    }

    let peakPressure = 0;
    let peakOp = null;
    const pressureAtOp = new Map();
    for (let i = 0; i < topo.length; i++) {
      const op = topo[i];
      const live = liveIn.get(op);
      let bytes = 0;
      for (const v of live) {
        if (v.type instanceof TensorType) {
          const s = v.type.sizeInBytes();
          if (s !== DYNAMIC) bytes += s;
        }
      }
      pressureAtOp.set(op, bytes);
      if (bytes > peakPressure) {
        peakPressure = bytes;
        peakOp = op;
      }
    }

    return new LivenessResult(
      liveIn, liveOut, intervals, opIndex,
      peakPressure, peakOp, pressureAtOp
    );
  }
}
