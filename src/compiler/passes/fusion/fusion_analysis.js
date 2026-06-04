import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { canInlineFuse, hasLoweringRule } from '../lowering/graph_to_tensor.js';

export const FusionKind = Object.freeze({
  ELEMENTWISE: 'kElementwise',
  BROADCAST: 'kBroadcast',
  REDUCTION: 'kReduction',
  INJECTIVE: 'kInjective',
  OPAQUE: 'kOpaque',
  HORIZONTAL: 'kHorizontal'
});

export function classifyFusionKind(ops) {
  let hasReduction = false;
  let hasInjective = false;
  let hasOpaque = false;
  let hasBroadcast = false;

  for (const op of ops) {
    const def = registry.get(op.opName);
    if (!def || def.isOpaque) { hasOpaque = true; continue; }
    if (def.isReduction) hasReduction = true;
    else if (def.isInjective) hasInjective = true;
    else if (def.isBroadcast) hasBroadcast = true;
    else if (!def.isElementwise) hasOpaque = true;
  }

  if (hasOpaque) return FusionKind.OPAQUE;
  if (hasReduction) return FusionKind.REDUCTION;
  if (hasInjective) return FusionKind.INJECTIVE;
  if (hasBroadcast) return FusionKind.BROADCAST;
  return FusionKind.ELEMENTWISE;
}

function classifyOp(def) {
  if (!def || def.isOpaque) return FusionKind.OPAQUE;
  if (def.isReduction) return FusionKind.REDUCTION;
  if (def.isInjective) return FusionKind.INJECTIVE;
  if (def.isBroadcast) return FusionKind.BROADCAST;
  if (def.isElementwise) return FusionKind.ELEMENTWISE;
  return FusionKind.OPAQUE;
}

const LEGAL_TRUE = Object.freeze({ legal: true });

export class FusionLegality {
  constructor(targetCapabilities = {}) {
    this.maxFusionSize = targetCapabilities.maxFusionSize || 512;
    this.maxSharedMemory = targetCapabilities.maxSharedMemory || 49152;
    this.libraryOps = targetCapabilities.libraryOps || new Set();
    this.allowReductionFusion = targetCapabilities.allowReductionFusion !== false;
    this._lowerableCache = new Map();
  }

  isOpLowerable(opName) {
    let result = this._lowerableCache.get(opName);
    if (result === undefined) {
      result = opName === 'return' || opName === 'yield' || canInlineFuse(opName) || hasLoweringRule(opName);
      this._lowerableCache.set(opName, result);
    }
    return result;
  }

  canFuse(producer, consumer) {
    if (!producer || !consumer) return { legal: false, reason: 'null op' };
    if (producer === consumer) return { legal: false, reason: 'same op' };

    if (producer.regions.length > 0 && producer.opName !== 'fusion' && producer.opName !== 'reduce') {
      return { legal: false, reason: 'producer has control flow regions' };
    }
    if (consumer.regions.length > 0 && consumer.opName !== 'fusion' && consumer.opName !== 'reduce') {
      return { legal: false, reason: 'consumer has control flow regions' };
    }

    const pDef = registry.get(producer.opName);
    const cDef = registry.get(consumer.opName);
    if (!pDef) return { legal: false, reason: 'unknown producer op' };
    if (!cDef) return { legal: false, reason: 'unknown consumer op' };

    if (!this.isOpLowerable(producer.opName)) {
      return { legal: false, reason: `producer op '${producer.opName}' has no lowering rule` };
    }
    if (!this.isOpLowerable(consumer.opName)) {
      return { legal: false, reason: `consumer op '${consumer.opName}' has no lowering rule` };
    }

    if (pDef.isOpaque) {
      return { legal: false, reason: 'producer is opaque (use EpilogueFusionPass for dot/conv epilogues)' };
    }
    if (cDef.isOpaque) {
      return { legal: false, reason: 'consumer is opaque' };
    }

    const pKind = classifyOp(pDef);
    const cKind = classifyOp(cDef);

    if (pKind === FusionKind.ELEMENTWISE && cKind === FusionKind.ELEMENTWISE) {
      return this._checkElementwisePair(producer, consumer);
    }

    if (pKind === FusionKind.BROADCAST && cKind === FusionKind.ELEMENTWISE) {
      return LEGAL_TRUE;
    }

    if (pKind === FusionKind.ELEMENTWISE && cKind === FusionKind.REDUCTION) {
      if (!this.allowReductionFusion) {
        return { legal: false, reason: 'reduction fusion disabled by target' };
      }
      return this._checkProducerConsumerShapes(producer, consumer);
    }

    if (pKind === FusionKind.REDUCTION && cKind === FusionKind.ELEMENTWISE) {
      if (!this.allowReductionFusion) {
        return { legal: false, reason: 'reduction fusion disabled by target' };
      }
      return LEGAL_TRUE;
    }

    if (pKind === FusionKind.REDUCTION && cKind === FusionKind.REDUCTION) {
      return { legal: false, reason: 'cannot fuse multiple reductions' };
    }

    if ((pKind === FusionKind.INJECTIVE && cKind === FusionKind.ELEMENTWISE) ||
        (pKind === FusionKind.ELEMENTWISE && cKind === FusionKind.INJECTIVE)) {
      return this._checkProducerConsumerShapes(producer, consumer);
    }

    return this._checkProducerConsumerShapes(producer, consumer);
  }

  canMergeGroups(groupA, groupB) {
    if (groupA.size + groupB.size > this.maxFusionSize) {
      return { legal: false, reason: 'merged group exceeds max fusion size' };
    }

    let reductionCount = 0;
    for (const op of groupA.ops) {
      const def = registry.get(op.opName);
      if (def && def.isReduction) reductionCount++;
      if (def && def.isOpaque) return { legal: false, reason: 'opaque op in merge (use EpilogueFusionPass)' };
    }
    for (const op of groupB.ops) {
      const def = registry.get(op.opName);
      if (def && def.isReduction) reductionCount++;
      if (def && def.isOpaque) return { legal: false, reason: 'opaque op in merge (use EpilogueFusionPass)' };
    }
    if (reductionCount > 1) {
      return { legal: false, reason: 'merged group would contain multiple reductions' };
    }

    return LEGAL_TRUE;
  }

  _checkElementwisePair(producer, consumer) {
    const pShape = this._getOutputShape(producer);
    const cShape = this._getOutputShape(consumer);
    if (pShape && cShape && !shapesCompatible(pShape, cShape)) {
      return { legal: false, reason: `elementwise shape mismatch: [${pShape}] vs [${cShape}]` };
    }
    return LEGAL_TRUE;
  }

  _checkProducerConsumerShapes(producer, consumer) {
    const pShape = this._getOutputShape(producer);
    if (!pShape) return LEGAL_TRUE;

    for (let i = 0; i < consumer.numOperands; i++) {
      if (consumer.getOperand(i).definingOp === producer) {
        const t = consumer.getOperand(i).type;
        if (t instanceof TensorType) {
          if (!shapesCompatible(pShape, t.shape)) {
            return { legal: false, reason: `shape mismatch on data edge: [${pShape}] vs [${t.shape}]` };
          }
          return LEGAL_TRUE;
        }
      }
    }
    return LEGAL_TRUE;
  }

  _getOutputShape(op) {
    for (let i = 0; i < op.numResults; i++) {
      const t = op.getResult(i).type;
      if (t instanceof TensorType) return t.shape;
    }
    return null;
  }
}

function shapesCompatible(shapeA, shapeB) {
  if (shapeA.length !== shapeB.length) return false;
  for (let i = 0; i < shapeA.length; i++) {
    if (shapeA[i] === DYNAMIC || shapeB[i] === DYNAMIC) continue;
    if (shapeA[i] !== shapeB[i]) return false;
  }
  return true;
}
