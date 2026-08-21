import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { FusionGroup } from './fusion_groups.js';

export type BenefitWeights = { memory: number; launch: number };
export type FusionCostBreakdown = {
  unfusedFLOPs: number;
  unfusedBytes: number;
  fusedFLOPs: number;
  fusedBytes: number;
  recomputeCost: number;
  memorySaved: number;
  launchSaved: number;
  registerPressure: number;
  sharedMemoryUsage: number;
  parallelismLoss: number;
  libraryCallLoss: number;
};
export type FusionDecision = { fuse: boolean; reason: string; cost?: FusionCostBreakdown };
export type FusionPolicy = { shouldFuse?(group: FusionGroup, model: FusionCostModel): FusionDecision | null };
export type FusionCostConfig = {
  memoryBandwidthGBs?: number;
  computeTFLOPs?: number;
  launchOverheadUs?: number;
  minBenefitRatio?: number;
  maxRegistersPerThread?: number;
  maxSharedMemory?: number;
  maxCodeSizeOps?: number;
  hasLibraryOp?: (opName: string) => boolean;
  registerBytesPerOp?: number;
  policy?: FusionPolicy | null;
  benefitWeights?: Partial<BenefitWeights>;
};
export type OpCostEstimate = { flops: number; bytes: number; arithmeticIntensity: number };

const DEFAULT_BENEFIT_WEIGHTS = Object.freeze({
  memory: 1,
  launch: 1000,
});

function deviceLimit(stated: number | undefined, whenUnspecified: number): number {
  if (stated === undefined) return whenUnspecified;
  return stated === 0 ? Infinity : stated;
}

export class FusionCostModel {
  memoryBandwidthGBs: number;
  computeTFLOPs: number;
  launchOverheadUs: number;
  minBenefitRatio: number;
  maxRegistersPerThread: number;
  maxSharedMemory: number;
  maxCodeSizeOps: number;
  hasLibraryOp: (opName: string) => boolean;
  registerBytesPerOp: number;
  policy: FusionPolicy | null;
  benefitWeights: BenefitWeights;

  constructor(config: FusionCostConfig = {}) {
    this.memoryBandwidthGBs = config.memoryBandwidthGBs || 900;
    this.computeTFLOPs = config.computeTFLOPs || 15;
    this.launchOverheadUs = config.launchOverheadUs || 5;
    this.minBenefitRatio = config.minBenefitRatio || 1.05;
    this.maxRegistersPerThread = deviceLimit(config.maxRegistersPerThread, 255);
    this.maxSharedMemory = deviceLimit(config.maxSharedMemory, 49152);
    this.maxCodeSizeOps = config.maxCodeSizeOps || 256;
    this.hasLibraryOp = config.hasLibraryOp || (() => false);
    this.registerBytesPerOp = config.registerBytesPerOp || 8;
    this.policy = config.policy || null;
    this.benefitWeights = { ...DEFAULT_BENEFIT_WEIGHTS, ...(config.benefitWeights || {}) };
  }

  edgeBenefit(bytes: number): number {
    const w = this.benefitWeights;
    return w.launch * this.launchOverheadUs + w.memory * bytes;
  }

  estimateOpCost(op: Operation): OpCostEstimate {
    const flops = this.estimateFLOPs(op);
    const bytes = this.estimateBytes(op);
    return { flops, bytes, arithmeticIntensity: bytes > 0 ? flops / bytes : 0 };
  }

  estimateFLOPs(op: Operation): number {
    const def = registry.get(op.opName);
    if (def && def.getFlops) return def.getFlops(op);

    let outputElements = 1;
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) {
        const n = t.numel();
        if (n !== DYNAMIC) outputElements = n as number;
        break;
      }
    }

    if (def && def.isReduction && op.numOperands > 0) {
      const t = op.getOperand(0).type;
      if (t instanceof TensorType) {
        const n = t.numel();
        if (n !== DYNAMIC) return n as number;
      }
    }

    return outputElements;
  }

  estimateBytes(op: Operation): number {
    let total = 0;
    for (let i = 0; i < op.numOperands; i++) {
      const t = op.getOperand(i).type;
      if (t instanceof TensorType) {
        const bytes = t.sizeInBytes();
        if (bytes !== DYNAMIC) total += bytes as number;
      }
    }
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) {
        const bytes = t.sizeInBytes();
        if (bytes !== DYNAMIC) total += bytes as number;
      }
    }
    return total;
  }

  estimateGroupCost(group: FusionGroup): FusionCostBreakdown {
    const opSet = group.opSet;
    const opFlops = new Map<Operation, number>();

    let totalFLOPs = 0;
    let totalBytes = 0;
    let recomputeCost = 0;
    let libraryCallLoss = 0;
    let hasReduction = false;
    let hasElementwise = false;
    let reductionElements = 0;
    let elementwiseElements = 0;
    let liveValues = 0;
    let maxLive = 0;
    const produced = new Set<Value>();

    for (const op of group.ops) {
      const flops = this.estimateFLOPs(op);
      opFlops.set(op, flops);
      totalFLOPs += flops;
      totalBytes += this.estimateBytes(op);

      if (this.hasLibraryOp(op.opName)) libraryCallLoss++;

      const def = registry.get(op.opName);
      if (def) {
        const elements = this._outputElements(op);
        if (def.isReduction) { hasReduction = true; reductionElements += elements; }
        if (def.isElementwise) { hasElementwise = true; elementwiseElements += elements; }
      }

      for (let i = 0; i < op.numResults; i++) {
        const res = op.getResult(i);
        produced.add(res);
        let internalUses = 0;
        let usedExternally = false;
        for (const use of res.uses()) {
          if (opSet.has(use.user)) internalUses++;
          else usedExternally = true;
        }
        if (internalUses > 1) {
          recomputeCost += (internalUses - 1) * flops;
        }
        if (internalUses > 0 || usedExternally) liveValues++;
      }

      for (let i = 0; i < op.numOperands; i++) {
        const operand = op.getOperand(i);
        if (!produced.has(operand)) continue;
        let stillNeeded = false;
        for (const use of operand.uses()) {
          if (opSet.has(use.user) && use.user !== op) {
            stillNeeded = true;
            break;
          }
        }
        if (!stillNeeded) liveValues--;
      }

      if (liveValues > maxLive) maxLive = liveValues;
    }

    const inputValues = group.getInputValues();
    const outputValues = group.getOutputValues();
    let fusedBytes = 0;
    for (const v of inputValues) {
      if (v.type instanceof TensorType) {
        const b = v.type.sizeInBytes();
        if (b !== DYNAMIC) fusedBytes += b as number;
      }
    }
    for (const v of outputValues) {
      if (v.type instanceof TensorType) {
        const b = v.type.sizeInBytes();
        if (b !== DYNAMIC) fusedBytes += b as number;
      }
    }

    let sharedMemoryUsage = 0;
    for (const op of group.ops) {
      for (let i = 0; i < op.numResults; i++) {
        const res = op.getResult(i);
        let internalUsers = 0;
        for (const use of res.uses()) {
          if (opSet.has(use.user)) {
            internalUsers++;
            if (internalUsers > 1) break;
          }
        }
        if (internalUsers > 1) {
          const t = res.type;
          if (t instanceof TensorType) {
            const bytes = t.sizeInBytes();
            if (bytes !== DYNAMIC) sharedMemoryUsage += bytes as number;
          }
        }
      }
    }

    let parallelismLoss = 0;
    if (hasReduction && hasElementwise && elementwiseElements > 0 && reductionElements > 0) {
      parallelismLoss = Math.abs(elementwiseElements - reductionElements);
    }

    return {
      unfusedFLOPs: totalFLOPs,
      unfusedBytes: totalBytes,
      fusedFLOPs: totalFLOPs + recomputeCost,
      fusedBytes,
      recomputeCost,
      memorySaved: totalBytes - fusedBytes,
      launchSaved: (group.size - 1) * this.launchOverheadUs,
      registerPressure: maxLive * this.registerBytesPerOp,
      sharedMemoryUsage,
      parallelismLoss,
      libraryCallLoss
    };
  }

  shouldFuse(group: FusionGroup): FusionDecision {
    if (group.size < 2) return { fuse: false, reason: 'group too small' };

    if (this.policy && typeof this.policy.shouldFuse === 'function') {
      const override = this.policy.shouldFuse(group, this);
      if (override) return override;
    }

    const cost = this.estimateGroupCost(group);

    if (cost.libraryCallLoss > 0) {
      return { fuse: false, reason: 'fusion would lose library call opportunity', cost };
    }

    if (cost.registerPressure > this.maxRegistersPerThread) {
      return { fuse: false, reason: `register pressure ${cost.registerPressure} exceeds limit ${this.maxRegistersPerThread}`, cost };
    }

    if (cost.sharedMemoryUsage > this.maxSharedMemory) {
      return { fuse: false, reason: `shared memory ${cost.sharedMemoryUsage} exceeds limit ${this.maxSharedMemory}`, cost };
    }

    if (group.size > this.maxCodeSizeOps) {
      return { fuse: false, reason: `code size ${group.size} exceeds limit ${this.maxCodeSizeOps}`, cost };
    }

    if (cost.memorySaved <= 0 && cost.launchSaved <= 0) {
      return { fuse: false, reason: 'no memory or launch benefit', cost };
    }

    if (cost.parallelismLoss > 0) {
      const benefit = cost.memorySaved + cost.launchSaved * 1000;
      if (cost.parallelismLoss > benefit * this.minBenefitRatio) {
        return { fuse: false, reason: 'parallelism loss outweighs fusion benefit', cost };
      }
    }

    return {
      fuse: true,
      reason: `saves ${cost.memorySaved} bytes, ${cost.launchSaved}us launch`,
      cost
    };
  }

  _outputElements(op: Operation): number {
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) {
        const n = t.numel();
        if (n !== DYNAMIC) return n as number;
      }
    }
    return 0;
  }
}
