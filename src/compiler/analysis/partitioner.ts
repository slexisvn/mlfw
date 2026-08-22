import { registry } from '../ir/graph/ops.js';
import { TensorType } from '../ir/graph/types.js';
import { buildPartitions } from '../passes/partition/partition_core.js';
import { OpGroup } from '../passes/partition/op_group.js';
import type { GraphFunction } from '../ir/graph/function.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';

export type PartitionTarget = {
  name: string;
  kind?: string;
  computeTFLOPs: number;
  hasLibraryOp(opName: string): boolean;
  isGPU(): boolean;
  isCPU(): boolean;
  isWasm(): boolean;
};

export type CostWeights = { transferCost: number; loadBalance: number };

export type PartitionerOpts = Readonly<{
  targets?: readonly PartitionTarget[];
  defaultTarget?: PartitionTarget | null;
  opTargetOverrides?: ReadonlyMap<string, PartitionTarget>;
  memoryLimits?: ReadonlyMap<string, number>;
  minPartitionSize?: number;
  costWeights?: Partial<CostWeights>;
}>;

export type TransferEdge = {
  src: Partition;
  dst: Partition;
  value: Value;
  sizeBytes: number;
};

export class Partition extends OpGroup {
  target: PartitionTarget;
  memoryBytes_?: number;
  private _memoryBytes: number;

  constructor(id: number, target: PartitionTarget) {
    super(id);
    this.target = target;
    this._memoryBytes = 0;
  }

  addOp(op: Operation): boolean {
    if (!super.addOp(op)) return false;
    this._memoryBytes += estimateOpMemory(op);
    return true;
  }

  merge(other: Partition): void {
    for (const op of other.ops) {
      this.addOp(op);
    }
  }

  get memoryBytes(): number {
    return this._memoryBytes;
  }
}

function estimateOpMemory(op: Operation): number {
  let bytes = 0;
  for (let i = 0; i < op.numResults; i++) {
    const type = op.getResult(i).type;
    if (type instanceof TensorType && type.isFullyStatic) {
      bytes += type.sizeInBytes();
    }
  }
  return bytes;
}

export class PartitionerConfig {
  targets: readonly PartitionTarget[];
  defaultTarget: PartitionTarget | null;
  opTargetOverrides: ReadonlyMap<string, PartitionTarget>;
  memoryLimits: ReadonlyMap<string, number>;
  minPartitionSize: number;
  costWeights: CostWeights;

  constructor(opts: PartitionerOpts = {}) {
    this.targets = opts.targets || [];
    this.defaultTarget = opts.defaultTarget || null;
    this.opTargetOverrides = opts.opTargetOverrides || new Map();
    this.memoryLimits = opts.memoryLimits || new Map();
    this.minPartitionSize = opts.minPartitionSize || 1;
    this.costWeights = {
      transferCost: 1.0,
      loadBalance: 0.5,
      ...(opts.costWeights || {}),
    };
  }
}

export class PartitionResult {
  partitions: Partition[];
  opToPartition: Map<Operation, Partition>;
  transferEdges: TransferEdge[];

  constructor(partitions: Partition[], opToPartition: Map<Operation, Partition>, transferEdges: TransferEdge[]) {
    this.partitions = partitions;
    this.opToPartition = opToPartition;
    this.transferEdges = transferEdges;
  }

  getPartition(op: Operation): Partition | null {
    return this.opToPartition.get(op) || null;
  }

  getPartitionsForTarget(target: PartitionTarget): Partition[] {
    return this.partitions.filter(p => p.target === target || p.target.name === target.name);
  }

  get numPartitions(): number {
    return this.partitions.length;
  }
}

export class GraphPartitioner {
  config: PartitionerConfig;
  private _supportCache: Map<PartitionTarget, Set<string>>;

  constructor(config: PartitionerConfig | PartitionerOpts) {
    this.config = config instanceof PartitionerConfig ? config : new PartitionerConfig(config);
    this._supportCache = new Map();
    this._buildSupportMap();
  }

  partition(func: GraphFunction): PartitionResult {
    const ops = this._collectPartitionableOps(func);
    const opToTarget = this._assignTargets(ops);
    const partitions = this._buildPartitions(ops, opToTarget);
    const merged = this._mergeSmallPartitions(partitions, opToTarget);
    const transferEdges = this._computeTransferEdges(merged);
    const opToPartition = new Map<Operation, Partition>();
    for (const p of merged) {
      for (const op of p.ops) {
        opToPartition.set(op, p);
      }
    }
    return new PartitionResult(merged, opToPartition, transferEdges);
  }

  _buildSupportMap(): void {
    for (const target of this.config.targets) {
      const supported = new Set<string>();
      for (const opName of registry.names()) {
        if (this._targetSupportsOp(target, opName)) {
          supported.add(opName);
        }
      }
      this._supportCache.set(target, supported);
    }
  }

  _targetSupportsOp(target: PartitionTarget, opName: string): boolean {
    if (target.hasLibraryOp(opName)) return true;

    const def = registry.get(opName);
    if (!def) return false;

    if (def.isConstant || def.isTerminator) return true;

    if (target.isGPU()) {
      return def.isElementwise || def.isReduction || def.isBroadcast ||
             def.isInjective || def.getAttr('gpuCapable') === true;
    }

    if (target.isCPU()) return true;

    if (target.isWasm()) {
      return !def.isOpaque && opName !== 'custom_call';
    }

    return false;
  }

  _collectPartitionableOps(func: GraphFunction): Operation[] {
    const ops: Operation[] = [];
    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      if (!def || def.isTerminator) continue;
      ops.push(op);
    }
    return ops;
  }

  _assignTargets(ops: readonly Operation[]): Map<Operation, PartitionTarget> {
    const opToTarget = new Map<Operation, PartitionTarget>();

    for (const op of ops) {
      const deviceAttr = op.getAttr<string>('device');
      if (deviceAttr) {
        const target = this._resolveDeviceAttr(deviceAttr);
        if (target) {
          opToTarget.set(op, target);
          continue;
        }
      }

      const override = this.config.opTargetOverrides.get(op.opName);
      if (override) {
        opToTarget.set(op, override);
        continue;
      }

      const target = this._selectBestTarget(op);
      opToTarget.set(op, target);
    }

    return opToTarget;
  }

  _resolveDeviceAttr(deviceAttr: string | PartitionTarget): PartitionTarget | null {
    if (typeof deviceAttr === 'string') {
      return this.config.targets.find(t => t.name === deviceAttr || t.kind === deviceAttr) || null;
    }
    return deviceAttr as PartitionTarget;
  }

  _selectBestTarget(op: Operation): PartitionTarget {
    let bestTarget = this.config.defaultTarget || this.config.targets[0];
    let bestScore = -Infinity;

    for (const target of this.config.targets) {
      const supported = this._supportCache.get(target);
      if (!supported || !supported.has(op.opName)) continue;

      const score = this._scoreTargetForOp(target, op);
      if (score > bestScore) {
        bestScore = score;
        bestTarget = target;
      }
    }

    return bestTarget;
  }

  _scoreTargetForOp(target: PartitionTarget, op: Operation): number {
    const def = registry.get(op.opName);
    if (!def) return 0;

    let score = 0;

    if (target.hasLibraryOp(op.opName)) {
      score += 100;
    }

    if (def.isReduction || def.isElementwise) {
      let totalElements = 0;
      for (let i = 0; i < op.numOperands; i++) {
        const type = op.getOperand(i).type;
        if (type instanceof TensorType && type.isFullyStatic) {
          totalElements += type.numel();
        }
      }

      if (target.isGPU() && totalElements > 1024) {
        score += 50;
      } else if (target.isCPU() && totalElements <= 1024) {
        score += 30;
      }
    }

    score += target.computeTFLOPs * 10;

    return score;
  }

  _buildPartitions(ops: readonly Operation[], opToTarget: ReadonlyMap<Operation, PartitionTarget>): Partition[] {
    const buildOpts = {
      sort: (xs: Operation[]) => this._topologicalSort(xs),
      labelOf: (op: Operation) => opToTarget.get(op),
      sameLabel: (a: PartitionTarget, b: PartitionTarget) => a === b || a.name === b.name,
      canMerge: (part: Partition, op: Operation, target: PartitionTarget) => this._fitsMemoryLimit(part, op, target),
      onAttach: (part: Partition, op: Operation) => { part.memoryBytes_ = (part.memoryBytes_ || 0) + estimateOpMemory(op); },
    };
    const { partitions } = (buildPartitions as unknown as (ops: readonly Operation[], opts: typeof buildOpts) => { partitions: { id: number; label: PartitionTarget; ops: Operation[] }[] })(ops, buildOpts);

    const byTargetName = new Map<string, Partition[]>();
    for (const p of partitions) {
      const partition = new Partition(p.id, p.label);
      for (const op of p.ops) partition.addOp(op);
      if (!byTargetName.has(p.label.name)) byTargetName.set(p.label.name, []);
      (byTargetName.get(p.label.name) as Partition[]).push(partition);
    }

    const allPartitions: Partition[] = [];
    for (const parts of byTargetName.values()) {
      for (const p of parts) {
        allPartitions.push(p);
      }
    }
    return allPartitions;
  }

  _fitsMemoryLimit(partition: Partition, op: Operation, target: PartitionTarget): boolean {
    const limit = this.config.memoryLimits.get(target.name);
    if (!limit) return true;
    const additionalMem = estimateOpMemory(op);
    return partition.memoryBytes + additionalMem <= limit;
  }

  _mergeSmallPartitions(partitions: Partition[], opToTarget: ReadonlyMap<Operation, PartitionTarget>): Partition[] {
    if (partitions.length <= 1) return partitions;

    const opToPart = new Map<Operation, Partition>();
    for (const part of partitions) for (const op of part.ops) opToPart.set(op, part);

    const mergedAway = new Set<Partition>();
    const succCache = new Map<Partition, Set<Partition>>();
    let reachCache = new Map<Partition, Set<Partition>>();
    for (const part of partitions) {
      if (mergedAway.has(part)) continue;
      succCache.set(part, new Set<Partition>());
    }
    for (const part of partitions) {
      if (mergedAway.has(part)) continue;
      const out = succCache.get(part) as Set<Partition>;
      for (const op of part.ops) {
        for (let r = 0; r < op.numResults; r++) {
          for (const use of op.getResult(r).uses()) {
            const cp = opToPart.get(use.user);
            if (cp && cp !== part) out.add(cp);
          }
        }
      }
    }

    const mergeSucc = (q: Partition, p: Partition): void => {
      const qSucc = succCache.get(q) as Set<Partition>;
      for (const s of succCache.get(p) as Set<Partition>) if (s !== q) qSucc.add(s);
      qSucc.delete(p);
      for (const [x, xs] of succCache) {
        if (xs.has(p)) {
          xs.delete(p);
          if (x !== q) xs.add(q);
        }
      }
      succCache.delete(p);
      reachCache = new Map();
    };

    const reachOf = (part: Partition): Set<Partition> => {
      let r = reachCache.get(part);
      if (r) return r;
      r = new Set<Partition>();
      const stack: Partition[] = [...(succCache.get(part) as Set<Partition>)];
      while (stack.length > 0) {
        const n = stack.pop() as Partition;
        if (r.has(n)) continue;
        r.add(n);
        const ns = succCache.get(n);
        if (ns) for (const s of ns) stack.push(s);
      }
      reachCache.set(part, r);
      return r;
    };
    const pathThroughIntermediate = (a: Partition, b: Partition): boolean => {
      for (const c of succCache.get(a) as Set<Partition>) {
        if (c !== b && reachOf(c).has(b)) return true;
      }
      return false;
    };
    const mergeCreatesCycle = (a: Partition, b: Partition): boolean => pathThroughIntermediate(a, b) || pathThroughIntermediate(b, a);

    const result: Partition[] = [];

    for (let i = 0; i < partitions.length; i++) {
      const p = partitions[i];
      if (mergedAway.has(p)) continue;

      if (p.size >= this.config.minPartitionSize) {
        result.push(p);
        continue;
      }

      let bestMerge = -1;
      let bestScore = -Infinity;

      for (let j = 0; j < partitions.length; j++) {
        if (i === j) continue;
        const candidate = partitions[j];
        if (mergedAway.has(candidate)) continue;
        if (candidate.target.name !== p.target.name) continue;
        if (mergeCreatesCycle(p, candidate)) continue;

        const score = this.config.costWeights.transferCost * this._mergeScore(p, candidate);
        if (score > bestScore) {
          bestScore = score;
          bestMerge = j;
        }
      }

      if (bestMerge >= 0) {
        const q = partitions[bestMerge];
        mergeSucc(q, p);
        q.merge(p);
        for (const op of p.ops) opToPart.set(op, q);
        mergedAway.add(p);
      } else {
        result.push(p);
      }
    }

    const covered = new Set<Operation>();
    for (const part of result) {
      for (const op of part.ops) covered.add(op);
    }
    for (const part of partitions) {
      let missing = false;
      for (const op of part.ops) {
        if (!covered.has(op)) { missing = true; break; }
      }
      if (missing) {
        result.push(part);
        for (const op of part.ops) covered.add(op);
      }
    }

    return result;
  }

  _mergeScore(a: Partition, b: Partition): number {
    let shared = 0;
    const aOutputs = new Set<Value>();
    for (const op of a.ops) {
      for (let i = 0; i < op.numResults; i++) {
        aOutputs.add(op.getResult(i));
      }
    }
    for (const op of b.ops) {
      for (let i = 0; i < op.numOperands; i++) {
        if (aOutputs.has(op.getOperand(i))) shared++;
      }
    }
    return shared;
  }

  _computeTransferEdges(partitions: readonly Partition[]): TransferEdge[] {
    const edges: TransferEdge[] = [];
    const seen = new Set<string>();
    const opToPartition = new Map<Operation, Partition>();
    for (const p of partitions) {
      for (const op of p.ops) {
        opToPartition.set(op, p);
      }
    }

    for (const p of partitions) {
      for (const op of p.ops) {
        for (let i = 0; i < op.numOperands; i++) {
          const producer = op.getOperand(i).definingOp;
          if (!producer) continue;
          const srcPartition = opToPartition.get(producer);
          if (!srcPartition || srcPartition === p) continue;

          const value = op.getOperand(i);
          const key = `${srcPartition.id}|${p.id}|${value.id}`;
          if (!seen.has(key)) {
            seen.add(key);
            edges.push({
              src: srcPartition,
              dst: p,
              value,
              sizeBytes: value.type instanceof TensorType && value.type.isFullyStatic
                ? value.type.sizeInBytes()
                : 0,
            });
          }
        }
      }
    }

    return edges;
  }

  _topologicalSort(ops: readonly Operation[]): Operation[] {
    const opSet = new Set(ops);
    const inDegree = new Map<Operation, number>();
    const adj = new Map<Operation, Operation[]>();

    for (const op of ops) {
      inDegree.set(op, 0);
      adj.set(op, []);
    }

    for (const op of ops) {
      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (producer && opSet.has(producer)) {
          (adj.get(producer) as Operation[]).push(op);
          inDegree.set(op, (inDegree.get(op) as number) + 1);
        }
      }
    }

    const queue: Operation[] = [];
    for (const op of ops) {
      if (inDegree.get(op) === 0) queue.push(op);
    }

    const sorted: Operation[] = [];
    let head = 0;
    while (head < queue.length) {
      const op = queue[head++];
      sorted.push(op);
      for (const consumer of adj.get(op) as Operation[]) {
        const deg = (inDegree.get(consumer) as number) - 1;
        inDegree.set(consumer, deg);
        if (deg === 0) queue.push(consumer);
      }
    }

    return sorted;
  }
}
