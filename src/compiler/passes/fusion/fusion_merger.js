import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { Block, Region } from '../../ir/graph/block.js';
import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { classifyFusionKind } from './fusion_analysis.js';
import {
  getYieldOp, countInnerOps, countReductions,
  allInnerOpsFusable, remapOperands
} from './fusion_utils.js';

export class FusionMergerPass extends FunctionPass {
  constructor(config = {}) {
    super('FusionMergerPass');
    this.maxFusionSize = config.maxFusionSize || 512;
    this.maxReductions = config.maxReductions || 1;
    this.launchOverheadUs = config.launchOverheadUs || 5;
    this.minMemorySavings = config.minMemorySavings || 0;
  }

  run(func) {
    const fusionOps = [];
    for (const op of func.ops()) {
      if (op.opName === 'fusion') fusionOps.push(op);
    }
    if (fusionOps.length < 2) return PassResult.UNCHANGED;

    const fusionSet = new Set(fusionOps);
    const edges = this._buildProducerConsumerEdges(fusionOps, fusionSet);
    if (edges.length === 0) return PassResult.UNCHANGED;

    let changed = false;
    const merged = new Set();

    for (const { producer, consumer, sharedResults } of edges) {
      if (merged.has(producer) || merged.has(consumer)) continue;
      if (!this._canMerge(producer, consumer)) continue;
      if (!this._shouldMerge(producer, consumer, sharedResults)) continue;
      if (this._mergeCreatesCycle(producer, consumer, fusionSet)) continue;

      this._merge(producer, consumer, sharedResults);
      merged.add(producer);
      changed = true;
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _buildProducerConsumerEdges(fusionOps, fusionSet) {
    const edges = [];
    const seen = new Map();

    for (const producer of fusionOps) {
      for (let r = 0; r < producer.numResults; r++) {
        const val = producer.getResult(r);
        for (const use of val.uses()) {
          if (!fusionSet.has(use.user)) continue;
          const consumer = use.user;
          if (consumer === producer) continue;

          const key = producer.id * 1000000 + consumer.id;
          let entry = seen.get(key);
          if (!entry) {
            entry = { producer, consumer, sharedResults: new Map() };
            seen.set(key, entry);
            edges.push(entry);
          }
          entry.sharedResults.set(r, use.operandIndex);
        }
      }
    }

    return edges;
  }

  _canMerge(producer, consumer) {
    const pInnerCount = countInnerOps(producer);
    const cInnerCount = countInnerOps(consumer);
    if (pInnerCount + cInnerCount > this.maxFusionSize) return false;

    let reductionCount = 0;
    reductionCount += countReductions(producer);
    reductionCount += countReductions(consumer);
    if (reductionCount > this.maxReductions) return false;

    if (!allInnerOpsFusable(producer)) return false;
    if (!allInnerOpsFusable(consumer)) return false;

    return true;
  }

  _shouldMerge(producer, consumer, sharedResults) {
    let intermediateBytes = 0;
    for (const [resultIdx] of sharedResults) {
      const val = producer.getResult(resultIdx);
      if (val.type instanceof TensorType) {
        const bytes = val.type.sizeInBytes();
        if (bytes !== DYNAMIC) intermediateBytes += bytes;
      }
    }

    let producerOnlyUsedByConsumer = true;
    for (let r = 0; r < producer.numResults; r++) {
      const val = producer.getResult(r);
      for (const use of val.uses()) {
        if (use.user !== consumer) {
          producerOnlyUsedByConsumer = false;
          break;
        }
      }
      if (!producerOnlyUsedByConsumer) break;
    }

    if (producerOnlyUsedByConsumer) return true;

    const recomputeFLOPs = this._estimateRecomputeCost(producer);
    const memBenefit = intermediateBytes + this.launchOverheadUs * 1000;
    return memBenefit > recomputeFLOPs;
  }

  _mergeCreatesCycle(producer, consumer, fusionSet) {
    const visited = new Set();
    const worklist = [];

    for (let i = 0; i < consumer.numOperands; i++) {
      const operand = consumer.getOperand(i);
      const defOp = operand.definingOp;
      if (defOp && defOp !== producer && defOp.opName !== 'constant') {
        if (!visited.has(defOp)) {
          visited.add(defOp);
          worklist.push(defOp);
        }
      }
    }

    let head = 0;
    while (head < worklist.length) {
      const op = worklist[head++];
      if (op === producer) return true;

      for (let i = 0; i < op.numOperands; i++) {
        const defOp = op.getOperand(i).definingOp;
        if (defOp && !visited.has(defOp)) {
          visited.add(defOp);
          worklist.push(defOp);
        }
      }
    }

    return false;
  }

  _merge(producer, consumer, sharedResults) {
    const pBlock = producer.regions[0].entryBlock;
    const cBlock = consumer.regions[0].entryBlock;

    const pYield = getYieldOp(pBlock);
    const cYield = getYieldOp(cBlock);
    if (!pYield || !cYield) return;

    const consumerArgFromProducer = new Map();
    for (const [prodResultIdx, consOperandIdx] of sharedResults) {
      consumerArgFromProducer.set(consOperandIdx, prodResultIdx);
    }

    const mergedOperands = [];
    const pArgRemap = new Map();

    for (let i = 0; i < producer.numOperands; i++) {
      pArgRemap.set(i, mergedOperands.length);
      mergedOperands.push(producer.getOperand(i));
    }

    const cArgRemap = new Map();
    for (let i = 0; i < consumer.numOperands; i++) {
      if (consumerArgFromProducer.has(i)) continue;
      const operand = consumer.getOperand(i);
      let existingIdx = -1;
      for (let j = 0; j < mergedOperands.length; j++) {
        if (mergedOperands[j] === operand) { existingIdx = j; break; }
      }
      if (existingIdx >= 0) {
        cArgRemap.set(i, existingIdx);
      } else {
        cArgRemap.set(i, mergedOperands.length);
        mergedOperands.push(operand);
      }
    }

    const mergedInputTypes = mergedOperands.map(v => v.type);
    const mergedRegion = new Region();
    const mergedBlock = new Block(mergedInputTypes);
    mergedRegion.addBlock(mergedBlock);

    const valueMap = new Map();

    for (let i = 0; i < pBlock.arguments.length; i++) {
      valueMap.set(pBlock.arguments[i], mergedBlock.arguments[pArgRemap.get(i)]);
    }

    for (const op of pBlock.ops()) {
      if (op.opName === 'yield') continue;
      const mappedOperands = remapOperands(op, valueMap);
      const resultTypes = op.results.map(r => r.type);
      const clonedRegions = op.regions.length > 0 ? op.regions.map(r => this._cloneRegionShallow(r)) : null;
      const cloned = new Operation(op.opName, mappedOperands, resultTypes, new Map(op.attributes), clonedRegions);
      mergedBlock.pushOp(cloned);
      for (let i = 0; i < op.numResults; i++) {
        valueMap.set(op.getResult(i), cloned.getResult(i));
      }
    }

    for (let i = 0; i < pYield.numOperands; i++) {
      const yieldVal = pYield.getOperand(i);
      const mappedYieldVal = valueMap.get(yieldVal) || yieldVal;
      for (const [consArgIdx, prodResultIdx] of consumerArgFromProducer) {
        if (prodResultIdx === i) {
          valueMap.set(cBlock.arguments[consArgIdx], mappedYieldVal);
        }
      }
    }

    for (let i = 0; i < cBlock.arguments.length; i++) {
      if (consumerArgFromProducer.has(i)) continue;
      const newIdx = cArgRemap.get(i);
      if (newIdx !== undefined) {
        valueMap.set(cBlock.arguments[i], mergedBlock.arguments[newIdx]);
      }
    }

    for (const op of cBlock.ops()) {
      if (op.opName === 'yield') continue;
      const mappedOperands = remapOperands(op, valueMap);
      const resultTypes = op.results.map(r => r.type);
      const clonedRegions = op.regions.length > 0 ? op.regions.map(r => this._cloneRegionShallow(r)) : null;
      const cloned = new Operation(op.opName, mappedOperands, resultTypes, new Map(op.attributes), clonedRegions);
      mergedBlock.pushOp(cloned);
      for (let i = 0; i < op.numResults; i++) {
        valueMap.set(op.getResult(i), cloned.getResult(i));
      }
    }

    const mergedYieldOperands = [];
    const mergedOutputTypes = [];

    let producerExternalResults = [];
    for (let r = 0; r < producer.numResults; r++) {
      let hasExternalUse = false;
      for (const use of producer.getResult(r).uses()) {
        if (use.user !== consumer) { hasExternalUse = true; break; }
      }
      if (hasExternalUse) {
        const pYieldVal = pYield.getOperand(r);
        const mapped = valueMap.get(pYieldVal) || pYieldVal;
        producerExternalResults.push({ resultIdx: r, value: mapped });
        mergedYieldOperands.push(mapped);
        mergedOutputTypes.push(producer.getResult(r).type);
      }
    }

    for (let r = 0; r < consumer.numResults; r++) {
      const cYieldVal = cYield.getOperand(r);
      const mapped = valueMap.get(cYieldVal) || cYieldVal;
      mergedYieldOperands.push(mapped);
      mergedOutputTypes.push(consumer.getResult(r).type);
    }

    const yieldOp = new Operation('yield', mergedYieldOperands, []);
    mergedBlock.pushOp(yieldOp);

    const mergedInnerOps = [];
    for (const op of mergedBlock.ops()) {
      if (op.opName !== 'yield') mergedInnerOps.push(op);
    }
    const mergedKind = classifyFusionKind(mergedInnerOps);

    const mergedFusionOp = new Operation(
      'fusion',
      mergedOperands,
      mergedOutputTypes,
      { fusion_kind: mergedKind },
      [mergedRegion]
    );

    const insertRef = consumer;
    if (!insertRef.parentBlock) return;
    insertRef.parentBlock.insertBefore(mergedFusionOp, insertRef);

    let outputIdx = 0;
    for (const { resultIdx } of producerExternalResults) {
      producer.getResult(resultIdx).replaceAllUsesWith(mergedFusionOp.getResult(outputIdx));
      outputIdx++;
    }
    for (let r = 0; r < consumer.numResults; r++) {
      consumer.getResult(r).replaceAllUsesWith(mergedFusionOp.getResult(outputIdx));
      outputIdx++;
    }

    producer.dropAllOperands();
    if (producer.parentBlock) producer.parentBlock.removeOp(producer);
    consumer.dropAllOperands();
    if (consumer.parentBlock) consumer.parentBlock.removeOp(consumer);
  }

  _estimateRecomputeCost(producer) {
    let flops = 0;
    const block = producer.regions[0]?.entryBlock;
    if (!block) return 0;
    for (const op of block.ops()) {
      if (op.opName === 'yield') continue;
      const def = registry.get(op.opName);
      if (def && def.getFlops) {
        flops += def.getFlops(op);
      } else {
        for (let i = 0; i < op.numResults; i++) {
          const t = op.getResult(i).type;
          if (t instanceof TensorType) {
            const n = t.numel();
            if (n !== DYNAMIC) flops += n;
            break;
          }
        }
      }
    }
    return flops;
  }
}
