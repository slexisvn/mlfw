import type { Operation } from './operation.js';
import type { Value } from './value.js';

export type CyclePolicy = 'ignore' | 'throw' | 'null';

type VisitState = 1 | 2;
type TopoFrame = { op: Operation; i: number };

function collectRegionDefs(op: Operation, defined: Set<Value>, inner: Operation[]): void {
  for (const region of op.regions) {
    for (const block of region) {
      for (const arg of block.arguments) defined.add(arg);
      for (const child of block) {
        inner.push(child);
        for (const result of child.results) defined.add(result);
        collectRegionDefs(child, defined, inner);
      }
    }
  }
}

export function capturedValues(op: Operation): Value[] {
  if (op.regions.length === 0) return [];
  const defined = new Set<Value>();
  const inner: Operation[] = [];
  collectRegionDefs(op, defined, inner);

  const captured: Value[] = [];
  const seen = new Set<Value>();
  for (const child of inner) {
    for (let i = 0; i < child.numOperands; i++) {
      const v = child.getOperand(i);
      if (defined.has(v) || seen.has(v)) continue;
      seen.add(v);
      captured.push(v);
    }
  }
  return captured;
}

export function readValues(op: Operation): Value[] {
  const values: Value[] = [];
  for (let i = 0; i < op.numOperands; i++) values.push(op.getOperand(i));
  for (const v of capturedValues(op)) values.push(v);
  return values;
}

export function topoSortByOperands(ops: Iterable<Operation>, contains: (op: Operation) => boolean, onCycle?: 'ignore' | 'throw'): Operation[];
export function topoSortByOperands(ops: Iterable<Operation>, contains: (op: Operation) => boolean, onCycle: 'null'): Operation[] | null;
export function topoSortByOperands(ops: Iterable<Operation>, contains: (op: Operation) => boolean, onCycle: CyclePolicy): Operation[] | null;
export function topoSortByOperands(ops: Iterable<Operation>, contains: (op: Operation) => boolean, onCycle: CyclePolicy = 'ignore'): Operation[] | null {
  const ordered: Operation[] = [];
  const state = new Map<Operation, VisitState>();
  const reads = new Map<Operation, readonly Value[]>();
  const readsOf = (op: Operation): readonly Value[] => {
    let r = reads.get(op);
    if (r === undefined) { r = readValues(op); reads.set(op, r); }
    return r;
  };
  for (const root of ops) {
    if (state.get(root) !== undefined) continue;
    state.set(root, 1);
    const stack: TopoFrame[] = [{ op: root, i: 0 }];
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const op = frame.op;
      const opReads = readsOf(op);
      if (frame.i < opReads.length) {
        const def = opReads[frame.i].definingOp;
        frame.i++;
        if (def && contains(def)) {
          const ds = state.get(def);
          if (ds === undefined) {
            state.set(def, 1);
            stack.push({ op: def, i: 0 });
          } else if (ds === 1) {
            if (onCycle === 'throw') throw new Error('topo sort: cycle detected');
            if (onCycle === 'null') return null;
          }
        }
        continue;
      }
      state.set(op, 2);
      ordered.push(op);
      stack.pop();
    }
  }
  return ordered;
}

export function topoSortOpSet(ops: Iterable<Operation>, onCycle?: 'ignore' | 'throw'): Operation[];
export function topoSortOpSet(ops: Iterable<Operation>, onCycle: 'null'): Operation[] | null;
export function topoSortOpSet(ops: Iterable<Operation>, onCycle: CyclePolicy): Operation[] | null;
export function topoSortOpSet(ops: Iterable<Operation>, onCycle: CyclePolicy = 'throw'): Operation[] | null {
  const arr = Array.isArray(ops) ? ops : [...ops];
  const set = new Set(arr);
  return topoSortByOperands(arr, (op) => set.has(op), onCycle);
}
