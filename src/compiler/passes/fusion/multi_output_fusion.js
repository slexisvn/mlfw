import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { Block, Region } from '../../ir/graph/block.js';
import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { classifyFusionKind } from './fusion_analysis.js';
import { TraceLevel } from '../../pipeline/trace.js';
import {
  getYieldOp, countInnerOps, countReductions,
  allInnerOpsFusable, remapOperands, makeComesBefore
} from './fusion_utils.js';

export class MultiOutputFusionPass extends FunctionPass {
  constructor(config = {}) {
    super('MultiOutputFusionPass');
    this.maxFusionSize = config.maxFusionSize || 512;
    this.maxReductions = config.maxReductions || 1;
    this.minSharedInputBytes = config.minSharedInputBytes || 0;
    this.maxOutputs = config.maxOutputs || 8;
    this.maxConsumersPerInput = config.maxConsumersPerInput || 64;
  }

  run(func) {
    const fusionOps = [];
    for (const op of func.ops()) {
      if (op.opName === 'fusion') fusionOps.push(op);
    }
    if (fusionOps.length < 2) return PassResult.UNCHANGED;

    const candidates = this._findCandidates(fusionOps);
    if (candidates.length === 0) return PassResult.UNCHANGED;

    let changed = false;
    let mergeCount = 0;
    const merged = new Set();

    for (const { left, right, sharedInputs, sharedBytes } of candidates) {
      if (merged.has(left) || merged.has(right)) continue;
      if (!this._canMerge(left, right)) continue;
      if (this._mergeCreatesCycle(left, right)) continue;

      this._mergeMultiOutput(left, right, sharedInputs);
      merged.add(left);
      merged.add(right);
      changed = true;
      mergeCount++;
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        fusionOps: fusionOps.length, candidates: candidates.length, mergeCount,
        level: TraceLevel.DEBUG,
      });
    }

    return changed ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _findCandidates(fusionOps) {
    const inputMap = new Map();

    for (const fop of fusionOps) {
      for (let i = 0; i < fop.numOperands; i++) {
        const operand = fop.getOperand(i);
        const key = operand.id;
        let consumers = inputMap.get(key);
        if (!consumers) {
          consumers = [];
          inputMap.set(key, consumers);
        }
        consumers.push(fop);
      }
    }

    const pairScores = new Map();

    for (const [inputId, consumers] of inputMap) {
      if (consumers.length < 2) continue;
      const lim = Math.min(consumers.length, this.maxConsumersPerInput);
      for (let i = 0; i < lim; i++) {
        for (let j = i + 1; j < lim; j++) {
          const a = consumers[i], b = consumers[j];
          if (a === b) continue;
          const key = pairKey(a, b);
          let entry = pairScores.get(key);
          if (!entry) {
            entry = { left: a, right: b, sharedInputs: new Set(), sharedBytes: 0 };
            pairScores.set(key, entry);
          }
          entry.sharedInputs.add(inputId);
        }
      }
    }

    const candidates = [];
    for (const entry of pairScores.values()) {
      let bytes = 0;
      for (const inputId of entry.sharedInputs) {
        const consumers = inputMap.get(inputId);
        if (!consumers || consumers.length === 0) continue;
        const fop = consumers[0];
        for (let i = 0; i < fop.numOperands; i++) {
          const operand = fop.getOperand(i);
          if (operand.id === inputId && operand.type instanceof TensorType) {
            const b = operand.type.sizeInBytes();
            if (b !== DYNAMIC) bytes += b;
            break;
          }
        }
      }
      entry.sharedBytes = bytes;
      if (bytes >= this.minSharedInputBytes) {
        candidates.push(entry);
      }
    }

    candidates.sort((a, b) => b.sharedBytes - a.sharedBytes);
    return candidates;
  }

  _canMerge(left, right) {
    if (this._hasProducerConsumerEdge(left, right)) return false;
    if (this._hasProducerConsumerEdge(right, left)) return false;

    const leftInnerCount = countInnerOps(left);
    const rightInnerCount = countInnerOps(right);
    if (leftInnerCount + rightInnerCount > this.maxFusionSize) return false;

    const totalOutputs = left.numResults + right.numResults;
    if (totalOutputs > this.maxOutputs) return false;

    let reductionCount = 0;
    reductionCount += countReductions(left);
    reductionCount += countReductions(right);
    if (reductionCount > this.maxReductions) return false;

    if (!allInnerOpsFusable(left)) return false;
    if (!allInnerOpsFusable(right)) return false;

    return true;
  }

  _mergeCreatesCycle(left, right) {
    const visited = new Set();
    const worklist = [];

    for (let i = 0; i < left.numResults; i++) {
      const val = left.getResult(i);
      for (const use of val.uses()) {
        if (use.user !== right && !visited.has(use.user)) {
          visited.add(use.user);
          worklist.push(use.user);
        }
      }
    }

    let head = 0;
    while (head < worklist.length) {
      const op = worklist[head++];
      if (op === right) return true;
      for (let i = 0; i < op.numResults; i++) {
        for (const use of op.getResult(i).uses()) {
          if (!visited.has(use.user)) {
            visited.add(use.user);
            worklist.push(use.user);
          }
        }
      }
    }

    visited.clear();
    worklist.length = 0;

    for (let i = 0; i < right.numResults; i++) {
      const val = right.getResult(i);
      for (const use of val.uses()) {
        if (use.user !== left && !visited.has(use.user)) {
          visited.add(use.user);
          worklist.push(use.user);
        }
      }
    }

    head = 0;
    while (head < worklist.length) {
      const op = worklist[head++];
      if (op === left) return true;
      for (let i = 0; i < op.numResults; i++) {
        for (const use of op.getResult(i).uses()) {
          if (!visited.has(use.user)) {
            visited.add(use.user);
            worklist.push(use.user);
          }
        }
      }
    }

    return false;
  }

  _mergeMultiOutput(left, right, sharedInputIds) {
    const lBlock = left.regions[0].entryBlock;
    const rBlock = right.regions[0].entryBlock;
    const lYield = getYieldOp(lBlock);
    const rYield = getYieldOp(rBlock);
    if (!lYield || !rYield) return;

    const mergedOperands = [];
    const operandDedup = new Map();

    const lArgRemap = new Map();
    for (let i = 0; i < left.numOperands; i++) {
      const operand = left.getOperand(i);
      const key = operand.id;
      let idx = operandDedup.get(key);
      if (idx === undefined) {
        idx = mergedOperands.length;
        operandDedup.set(key, idx);
        mergedOperands.push(operand);
      }
      lArgRemap.set(i, idx);
    }

    const rArgRemap = new Map();
    for (let i = 0; i < right.numOperands; i++) {
      const operand = right.getOperand(i);
      const key = operand.id;
      let idx = operandDedup.get(key);
      if (idx === undefined) {
        idx = mergedOperands.length;
        operandDedup.set(key, idx);
        mergedOperands.push(operand);
      }
      rArgRemap.set(i, idx);
    }

    const mergedInputTypes = mergedOperands.map(v => v.type);
    const mergedRegion = new Region();
    const mergedBlock = new Block(mergedInputTypes);
    mergedRegion.addBlock(mergedBlock);

    const valueMap = new Map();

    for (let i = 0; i < lBlock.arguments.length; i++) {
      valueMap.set(lBlock.arguments[i], mergedBlock.arguments[lArgRemap.get(i)]);
    }

    for (const op of lBlock.ops()) {
      if (op.opName === 'yield') continue;
      const mappedOperands = remapOperands(op, valueMap);
      const resultTypes = op.results.map(r => r.type);
      const cloned = new Operation(op.opName, mappedOperands, resultTypes, new Map(op.attributes));
      mergedBlock.pushOp(cloned);
      for (let i = 0; i < op.numResults; i++) {
        valueMap.set(op.getResult(i), cloned.getResult(i));
      }
    }

    for (let i = 0; i < rBlock.arguments.length; i++) {
      valueMap.set(rBlock.arguments[i], mergedBlock.arguments[rArgRemap.get(i)]);
    }

    for (const op of rBlock.ops()) {
      if (op.opName === 'yield') continue;
      const mappedOperands = remapOperands(op, valueMap);
      const resultTypes = op.results.map(r => r.type);
      const cloned = new Operation(op.opName, mappedOperands, resultTypes, new Map(op.attributes));
      mergedBlock.pushOp(cloned);
      for (let i = 0; i < op.numResults; i++) {
        valueMap.set(op.getResult(i), cloned.getResult(i));
      }
    }

    const mergedYieldOperands = [];
    const mergedOutputTypes = [];

    for (let i = 0; i < lYield.numOperands; i++) {
      const orig = lYield.getOperand(i);
      mergedYieldOperands.push(valueMap.get(orig) || orig);
      mergedOutputTypes.push(left.getResult(i).type);
    }
    for (let i = 0; i < rYield.numOperands; i++) {
      const orig = rYield.getOperand(i);
      mergedYieldOperands.push(valueMap.get(orig) || orig);
      mergedOutputTypes.push(right.getResult(i).type);
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

    const block = left.parentBlock;
    if (!block) return;

    const comesBefore = makeComesBefore(block);
    let insertAfter = null;
    for (const val of mergedOperands) {
      const producer = val.definingOp;
      if (!producer || producer === left || producer === right) continue;
      if (!insertAfter || !comesBefore(producer, insertAfter)) {
        insertAfter = producer;
      }
    }

    if (insertAfter && insertAfter.parentBlock === block) {
      block.insertAfter(mergedFusionOp, insertAfter);
    } else {
      block.insertBefore(mergedFusionOp, left);
    }

    for (let i = 0; i < left.numResults; i++) {
      left.getResult(i).replaceAllUsesWith(mergedFusionOp.getResult(i));
    }
    const offset = left.numResults;
    for (let i = 0; i < right.numResults; i++) {
      right.getResult(i).replaceAllUsesWith(mergedFusionOp.getResult(offset + i));
    }

    left.dropAllOperands();
    if (left.parentBlock) left.parentBlock.removeOp(left);
    right.dropAllOperands();
    if (right.parentBlock) right.parentBlock.removeOp(right);
  }

  _hasProducerConsumerEdge(producer, consumer) {
    for (let r = 0; r < producer.numResults; r++) {
      for (const use of producer.getResult(r).uses()) {
        if (use.user === consumer) return true;
      }
    }
    return false;
  }

}

function pairKey(a, b) {
  const lo = Math.min(a.id, b.id);
  const hi = Math.max(a.id, b.id);
  return `${lo}|${hi}`;
}
