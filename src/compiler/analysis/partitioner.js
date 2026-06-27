import { registry } from '../ir/graph/ops.js';
import { TensorType } from '../ir/graph/types.js';
import { buildPartitions } from '../passes/partition/partition_core.js';

export class Partition {
  constructor(id, target) {
    this.id = id;
    this.target = target;
    this.ops = [];
    this.opSet = new Set();
    this._inputValues = null;
    this._outputValues = null;
    this._memoryBytes = 0;
  }

  addOp(op) {
    if (this.opSet.has(op)) return;
    this.ops.push(op);
    this.opSet.add(op);
    this._inputValues = null;
    this._outputValues = null;
    this._memoryBytes += estimateOpMemory(op);
  }

  hasOp(op) {
    return this.opSet.has(op);
  }

  merge(other) {
    for (const op of other.ops) {
      this.addOp(op);
    }
  }

  computeIO() {
    if (this._inputValues && this._outputValues) return;
    this._inputValues = [];
    this._outputValues = [];
    const inputSet = new Set();
    const outputSet = new Set();

    for (const op of this.ops) {
      for (let i = 0; i < op.numOperands; i++) {
        const operand = op.getOperand(i);
        if (operand.definingOp && this.opSet.has(operand.definingOp)) continue;
        if (!inputSet.has(operand)) {
          inputSet.add(operand);
          this._inputValues.push(operand);
        }
      }
      for (let i = 0; i < op.numResults; i++) {
        const result = op.getResult(i);
        if (outputSet.has(result)) continue;
        for (const use of result.uses()) {
          if (!this.opSet.has(use.user)) {
            outputSet.add(result);
            this._outputValues.push(result);
            break;
          }
        }
      }
    }
  }

  getInputValues() {
    this.computeIO();
    return this._inputValues;
  }

  getOutputValues() {
    this.computeIO();
    return this._outputValues;
  }

  get size() {
    return this.ops.length;
  }

  get memoryBytes() {
    return this._memoryBytes;
  }
}

function estimateOpMemory(op) {
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
  constructor(opts = {}) {
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
  constructor(partitions, opToPartition, transferEdges) {
    this.partitions = partitions;
    this.opToPartition = opToPartition;
    this.transferEdges = transferEdges;
  }

  getPartition(op) {
    return this.opToPartition.get(op) || null;
  }

  getPartitionsForTarget(target) {
    return this.partitions.filter(p => p.target === target || p.target.name === target.name);
  }

  get numPartitions() {
    return this.partitions.length;
  }
}

export class GraphPartitioner {
  constructor(config) {
    this.config = config instanceof PartitionerConfig ? config : new PartitionerConfig(config);
    this._supportCache = new Map();
    this._buildSupportMap();
  }

  partition(func) {
    const ops = this._collectPartitionableOps(func);
    const opToTarget = this._assignTargets(ops);
    const partitions = this._buildPartitions(ops, opToTarget);
    const merged = this._mergeSmallPartitions(partitions, opToTarget);
    const transferEdges = this._computeTransferEdges(merged);
    const opToPartition = new Map();
    for (const p of merged) {
      for (const op of p.ops) {
        opToPartition.set(op, p);
      }
    }
    return new PartitionResult(merged, opToPartition, transferEdges);
  }

  _buildSupportMap() {
    for (const target of this.config.targets) {
      const supported = new Set();
      for (const opName of registry.names()) {
        if (this._targetSupportsOp(target, opName)) {
          supported.add(opName);
        }
      }
      this._supportCache.set(target, supported);
    }
  }

  _targetSupportsOp(target, opName) {
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

  _collectPartitionableOps(func) {
    const ops = [];
    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      if (!def || def.isTerminator) continue;
      ops.push(op);
    }
    return ops;
  }

  _assignTargets(ops) {
    const opToTarget = new Map();

    for (const op of ops) {
      const deviceAttr = op.getAttr('device');
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

  _resolveDeviceAttr(deviceAttr) {
    if (typeof deviceAttr === 'string') {
      return this.config.targets.find(t => t.name === deviceAttr || t.kind === deviceAttr) || null;
    }
    return deviceAttr;
  }

  _selectBestTarget(op) {
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

  _scoreTargetForOp(target, op) {
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

  _buildPartitions(ops, opToTarget) {
    const { partitions } = buildPartitions(ops, {
      sort: (xs) => this._topologicalSort(xs),
      labelOf: (op) => opToTarget.get(op),
      sameLabel: (a, b) => a === b || a.name === b.name,
      canMerge: (part, op, target) => this._fitsMemoryLimit(part, op, target),
      onAttach: (part, op) => { part.memoryBytes = (part.memoryBytes || 0) + estimateOpMemory(op); },
    });

    const byTargetName = new Map();
    for (const p of partitions) {
      const partition = new Partition(p.id, p.label);
      for (const op of p.ops) partition.addOp(op);
      if (!byTargetName.has(p.label.name)) byTargetName.set(p.label.name, []);
      byTargetName.get(p.label.name).push(partition);
    }

    const allPartitions = [];
    for (const parts of byTargetName.values()) {
      for (const p of parts) {
        allPartitions.push(p);
      }
    }
    return allPartitions;
  }

  _fitsMemoryLimit(partition, op, target) {
    const limit = this.config.memoryLimits.get(target.name);
    if (!limit) return true;
    const additionalMem = estimateOpMemory(op);
    return partition.memoryBytes + additionalMem <= limit;
  }

  _mergeSmallPartitions(partitions, opToTarget) {
    if (partitions.length <= 1) return partitions;

    const opToPart = new Map();
    for (const part of partitions) for (const op of part.ops) opToPart.set(op, part);

    const mergedAway = new Set();
    let succCache = null;
    let reachCache = null;
    const buildSucc = () => {
      succCache = new Map();
      reachCache = new Map();
      for (const part of partitions) {
        if (mergedAway.has(part)) continue;
        succCache.set(part, new Set());
      }
      for (const part of partitions) {
        if (mergedAway.has(part)) continue;
        const out = succCache.get(part);
        for (const op of part.ops) {
          for (let r = 0; r < op.numResults; r++) {
            for (const use of op.getResult(r).uses()) {
              const cp = opToPart.get(use.user);
              if (cp && cp !== part) out.add(cp);
            }
          }
        }
      }
    };
    buildSucc();

    const reachOf = (part) => {
      let r = reachCache.get(part);
      if (r) return r;
      r = new Set();
      const stack = [...succCache.get(part)];
      while (stack.length > 0) {
        const n = stack.pop();
        if (r.has(n)) continue;
        r.add(n);
        for (const s of succCache.get(n)) stack.push(s);
      }
      reachCache.set(part, r);
      return r;
    };
    const pathThroughIntermediate = (a, b) => {
      for (const c of succCache.get(a)) {
        if (c !== b && reachOf(c).has(b)) return true;
      }
      return false;
    };
    const mergeCreatesCycle = (a, b) => pathThroughIntermediate(a, b) || pathThroughIntermediate(b, a);

    const result = [];

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

        const score = this._mergeScore(p, candidate);
        if (score > bestScore) {
          bestScore = score;
          bestMerge = j;
        }
      }

      if (bestMerge >= 0) {
        partitions[bestMerge].merge(p);
        for (const op of p.ops) opToPart.set(op, partitions[bestMerge]);
        mergedAway.add(p);
        buildSucc();
      } else {
        result.push(p);
      }
    }

    const covered = new Set();
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

  _mergeScore(a, b) {
    let shared = 0;
    const aOutputs = new Set();
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

  _computeTransferEdges(partitions) {
    const edges = [];
    const seen = new Set();
    const opToPartition = new Map();
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

  _topologicalSort(ops) {
    const opSet = new Set(ops);
    const inDegree = new Map();
    const adj = new Map();

    for (const op of ops) {
      inDegree.set(op, 0);
      adj.set(op, []);
    }

    for (const op of ops) {
      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (producer && opSet.has(producer)) {
          adj.get(producer).push(op);
          inDegree.set(op, inDegree.get(op) + 1);
        }
      }
    }

    const queue = [];
    for (const op of ops) {
      if (inDegree.get(op) === 0) queue.push(op);
    }

    const sorted = [];
    let head = 0;
    while (head < queue.length) {
      const op = queue[head++];
      sorted.push(op);
      for (const consumer of adj.get(op)) {
        const deg = inDegree.get(consumer) - 1;
        inDegree.set(consumer, deg);
        if (deg === 0) queue.push(consumer);
      }
    }

    return sorted;
  }
}
