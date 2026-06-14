import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';

export class FusionCostModel {
  constructor(config = {}) {
    this.memoryBandwidthGBs = config.memoryBandwidthGBs || 900;
    this.computeTFLOPs = config.computeTFLOPs || 15;
    this.launchOverheadUs = config.launchOverheadUs || 5;
    this.minBenefitRatio = config.minBenefitRatio || 1.05;
    this.maxRegistersPerThread = config.maxRegistersPerThread || 255;
    this.maxSharedMemory = config.maxSharedMemory || 49152;
    this.maxCodeSizeOps = config.maxCodeSizeOps || 256;
    this.libraryOps = config.libraryOps || new Set();
    this.registerBytesPerOp = config.registerBytesPerOp || 8;
  }

  estimateOpCost(op) {
    const flops = this.estimateFLOPs(op);
    const bytes = this.estimateBytes(op);
    return { flops, bytes, arithmeticIntensity: bytes > 0 ? flops / bytes : 0 };
  }

  estimateFLOPs(op) {
    const def = registry.get(op.opName);
    if (def && def.getFlops) return def.getFlops(op);

    let outputElements = 1;
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) {
        const n = t.numel();
        if (n !== DYNAMIC) outputElements = n;
        break;
      }
    }

    if (def && def.isReduction && op.numOperands > 0) {
      const t = op.getOperand(0).type;
      if (t instanceof TensorType) {
        const n = t.numel();
        if (n !== DYNAMIC) return n;
      }
    }

    return outputElements;
  }

  estimateBytes(op) {
    let total = 0;
    for (let i = 0; i < op.numOperands; i++) {
      const t = op.getOperand(i).type;
      if (t instanceof TensorType) {
        const bytes = t.sizeInBytes();
        if (bytes !== DYNAMIC) total += bytes;
      }
    }
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) {
        const bytes = t.sizeInBytes();
        if (bytes !== DYNAMIC) total += bytes;
      }
    }
    return total;
  }

  estimateGroupCost(group) {
    const opSet = group.opSet;
    const opFlops = new Map();

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
    const produced = new Set();

    for (const op of group.ops) {
      const flops = this.estimateFLOPs(op);
      opFlops.set(op, flops);
      totalFLOPs += flops;
      totalBytes += this.estimateBytes(op);

      if (this.libraryOps.has(op.opName)) libraryCallLoss++;

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
        if (b !== DYNAMIC) fusedBytes += b;
      }
    }
    for (const v of outputValues) {
      if (v.type instanceof TensorType) {
        const b = v.type.sizeInBytes();
        if (b !== DYNAMIC) fusedBytes += b;
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
            if (bytes !== DYNAMIC) sharedMemoryUsage += bytes;
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

  shouldFuse(group) {
    if (group.size < 2) return { fuse: false, reason: 'group too small' };

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

  _outputElements(op) {
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) {
        const n = t.numel();
        if (n !== DYNAMIC) return n;
      }
    }
    return 0;
  }
}
