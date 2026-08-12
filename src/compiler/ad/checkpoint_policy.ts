import { TensorType } from '../ir/graph/types.js';
import type { Operation } from '../ir/graph/operation.js';
import type { GraphFunction } from '../ir/graph/function.js';

export type BoundaryPredicate = (op: Operation) => boolean;

export class Segment {
  ops: Operation[];
  boundaryInputs: Set<number>;
  boundaryOutputs: Set<number>;

  constructor(ops: Operation[]) {
    this.ops = ops;
    this.boundaryInputs = new Set();
    this.boundaryOutputs = new Set();
  }
}

function computeBoundaries(segments: readonly Segment[]): void {
  const opToSegment = new Map<number, number>();
  for (let s = 0; s < segments.length; s++) {
    for (const op of segments[s].ops) {
      opToSegment.set(op.id, s);
      for (let r = 0; r < op.numResults; r++) {
        opToSegment.set(op.getResult(r).id, s);
      }
    }
  }

  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s];
    const localValueIds = new Set<number>();
    for (const op of seg.ops) {
      for (let r = 0; r < op.numResults; r++) {
        localValueIds.add(op.getResult(r).id);
      }
    }

    for (const op of seg.ops) {
      for (let o = 0; o < op.numOperands; o++) {
        const val = op.getOperand(o);
        if (!localValueIds.has(val.id)) {
          seg.boundaryInputs.add(val.id);
        }
      }
    }

    for (const op of seg.ops) {
      for (let r = 0; r < op.numResults; r++) {
        const result = op.getResult(r);
        for (const use of result.uses()) {
          const userOp = use.user;
          const userSeg = opToSegment.get(userOp.id);
          if (userSeg !== undefined && userSeg !== s) {
            seg.boundaryOutputs.add(result.id);
            break;
          }
        }
      }
    }
  }
}

function partitionOps(ops: readonly Operation[], segmentSize: number): Segment[] {
  const segments: Segment[] = [];
  for (let i = 0; i < ops.length; i += segmentSize) {
    segments.push(new Segment(ops.slice(i, i + segmentSize)));
  }
  return segments;
}

export class CheckpointPolicy {
  segment(_topoOrder: readonly Operation[], _forwardFunc?: GraphFunction): Segment[] {
    throw new Error('subclass must implement segment()');
  }
}

export class EveryKPolicy extends CheckpointPolicy {
  private _segmentSize: number;

  constructor({ segmentSize }: { segmentSize: number }) {
    super();
    this._segmentSize = segmentSize;
  }

  override segment(topoOrder: readonly Operation[]): Segment[] {
    const ops = topoOrder.filter(op => op.opName !== 'return' && op.opName !== 'constant');
    const segments = partitionOps(ops, this._segmentSize);
    computeBoundaries(segments);
    return segments;
  }
}

export class SqrtPolicy extends CheckpointPolicy {
  override segment(topoOrder: readonly Operation[]): Segment[] {
    const ops = topoOrder.filter(op => op.opName !== 'return' && op.opName !== 'constant');
    const segmentSize = Math.max(1, Math.ceil(Math.sqrt(ops.length)));
    const segments = partitionOps(ops, segmentSize);
    computeBoundaries(segments);
    return segments;
  }
}

export class MemoryBudgetPolicy extends CheckpointPolicy {
  private _threshold: number;

  constructor({ threshold }: { threshold: number }) {
    super();
    this._threshold = threshold;
  }

  override segment(topoOrder: readonly Operation[]): Segment[] {
    const ops = topoOrder.filter(op => op.opName !== 'return' && op.opName !== 'constant');
    const segments: Segment[] = [];
    let current: Operation[] = [];
    let currentBytes = 0;

    for (const op of ops) {
      let opBytes = 0;
      for (let r = 0; r < op.numResults; r++) {
        const type = op.getResult(r).type as TensorType;
        if (type && typeof type.sizeInBytes === 'function') {
          const size = type.sizeInBytes();
          if (size > 0) opBytes += size;
        }
      }

      if (current.length > 0 && currentBytes + opBytes > this._threshold) {
        segments.push(new Segment(current));
        current = [];
        currentBytes = 0;
      }

      current.push(op);
      currentBytes += opBytes;
    }

    if (current.length > 0) {
      segments.push(new Segment(current));
    }

    computeBoundaries(segments);
    return segments;
  }
}

export class ExplicitPolicy extends CheckpointPolicy {
  private _boundaries: BoundaryPredicate | Set<number>;

  constructor({ boundaries }: { boundaries: BoundaryPredicate | Iterable<number> }) {
    super();
    this._boundaries = typeof boundaries === 'function' ? boundaries : new Set(boundaries);
  }

  override segment(topoOrder: readonly Operation[]): Segment[] {
    const ops = topoOrder.filter(op => op.opName !== 'return' && op.opName !== 'constant');
    const isBoundary = typeof this._boundaries === 'function'
      ? this._boundaries
      : (op: Operation) => (this._boundaries as Set<number>).has(op.id);

    const segments: Segment[] = [];
    let current: Operation[] = [];

    for (const op of ops) {
      current.push(op);
      if (isBoundary(op) && current.length > 0) {
        segments.push(new Segment(current));
        current = [];
      }
    }

    if (current.length > 0) {
      segments.push(new Segment(current));
    }

    computeBoundaries(segments);
    return segments;
  }
}
