import { TensorType } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { classifyFusionKind, FusionKind } from './fusion_analysis.js';
import { canInlineFuse } from '../lowering/graph_to_tensor.js';

export class FusionGroup {
  constructor(id) {
    this.id = id;
    this.ops = [];
    this.opSet = new Set();
    this.kind = null;
    this._inputValues = null;
    this._outputValues = null;
  }

  addOp(op) {
    if (this.opSet.has(op)) return;
    this.ops.push(op);
    this.opSet.add(op);
    this._inputValues = null;
    this._outputValues = null;
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

  get size() { return this.ops.length; }

  classifyKind() {
    this.kind = classifyFusionKind(this.ops);
    return this.kind;
  }

  allOpsInlineFusable() {
    for (const op of this.ops) {
      const def = registry.get(op.opName);
      if (!def) return false;
      if (def.isReduction || def.isConstant) continue;
      if (!canInlineFuse(op.opName)) return false;
    }
    return true;
  }
}

function outputShapeKey(op) {
  for (let i = 0; i < op.numResults; i++) {
    const t = op.getResult(i).type;
    if (t instanceof TensorType) return t.shape.join(',');
  }
  return null;
}

export class FusionGroupBuilder {
  constructor(legality) {
    this.legality = legality;
    this._nextId = 0;
  }

  buildProducerConsumerGroups(func) {
    const groups = [];
    const opToGroup = new Map();

    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      if (!def || def.isConstant || def.isTerminator) continue;

      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (!producer) continue;
        const pDef = registry.get(producer.opName);
        if (!pDef || pDef.isConstant) continue;

        const consumerGroup = opToGroup.get(op);
        const producerGroup = opToGroup.get(producer);
        if (consumerGroup && consumerGroup === producerGroup) continue;

        const result = this.legality.canFuse(producer, op);
        if (!result.legal) continue;

        if (!consumerGroup && !producerGroup) {
          const group = new FusionGroup(this._nextId++);
          group.addOp(producer);
          group.addOp(op);
          opToGroup.set(producer, group);
          opToGroup.set(op, group);
        } else if (consumerGroup && !producerGroup) {
          if (consumerGroup.size < this.legality.maxFusionSize) {
            consumerGroup.addOp(producer);
            opToGroup.set(producer, consumerGroup);
          }
        } else if (!consumerGroup && producerGroup) {
          if (producerGroup.size < this.legality.maxFusionSize) {
            producerGroup.addOp(op);
            opToGroup.set(op, producerGroup);
          }
        } else {
          const mergeResult = this.legality.canMergeGroups(consumerGroup, producerGroup);
          if (mergeResult.legal) {
            consumerGroup.merge(producerGroup);
            for (const pOp of producerGroup.ops) {
              opToGroup.set(pOp, consumerGroup);
            }
          }
        }
      }
    }

    const seen = new Set();
    for (const group of opToGroup.values()) {
      if (seen.has(group) || group.size < 2) continue;
      seen.add(group);
      group.classifyKind();
      groups.push(group);
    }
    return groups;
  }

  buildHorizontalGroups(func) {
    const groups = [];
    const opToGroup = new Map();

    const shapeBuckets = new Map();
    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      if (!def || def.isConstant || def.isTerminator || def.isOpaque) continue;
      const key = outputShapeKey(op);
      if (key === null) continue;
      let bucket = shapeBuckets.get(key);
      if (!bucket) {
        bucket = [];
        shapeBuckets.set(key, bucket);
      }
      bucket.push(op);
    }

    for (const [, bucket] of shapeBuckets) {
      if (bucket.length < 2) continue;

      for (let i = 0; i < bucket.length; i++) {
        const op1 = bucket[i];
        if (opToGroup.has(op1)) continue;

        const group = new FusionGroup(this._nextId++);
        group.addOp(op1);
        opToGroup.set(op1, group);

        const def1 = registry.get(op1.opName);
        if (!def1 || op1.numResults === 0) continue;
        const type1 = op1.getResult(0).type;

        for (let j = i + 1; j < bucket.length; j++) {
          const op2 = bucket[j];
          if (opToGroup.has(op2)) continue;
          if (op2.numResults === 0) continue;
          if (!type1.equals(op2.getResult(0).type)) continue;
          if (this._hasDependency(op1, op2) || this._hasDependency(op2, op1)) continue;

          const def2 = registry.get(op2.opName);
          if (this._sharesInput(op1, op2) || (def1.isElementwise && def2 && def2.isElementwise)) {
            group.addOp(op2);
            opToGroup.set(op2, group);
          }
        }

        if (group.size >= 2) {
          group.kind = FusionKind.HORIZONTAL;
          groups.push(group);
        }
      }
    }

    return groups;
  }

  buildAllGroups(func) {
    const pcGroups = this.buildProducerConsumerGroups(func);
    const horizontalGroups = this.buildHorizontalGroups(func);

    const fusedOps = new Set();
    for (const g of pcGroups) {
      for (const op of g.ops) fusedOps.add(op);
    }

    const result = [...pcGroups];
    for (const h of horizontalGroups) {
      let overlaps = false;
      for (const op of h.ops) {
        if (fusedOps.has(op)) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) {
        result.push(h);
        for (const op of h.ops) fusedOps.add(op);
      }
    }

    return result;
  }

  _sharesInput(op1, op2) {
    if (op1.numOperands === 0 || op2.numOperands === 0) return false;
    const op1Operands = new Set();
    for (let k = 0; k < op1.numOperands; k++) {
      op1Operands.add(op1.getOperand(k));
    }
    for (let l = 0; l < op2.numOperands; l++) {
      if (op1Operands.has(op2.getOperand(l))) return true;
    }
    return false;
  }

  _hasDependency(source, target) {
    for (let k = 0; k < source.numResults; k++) {
      const res = source.getResult(k);
      for (let l = 0; l < target.numOperands; l++) {
        if (target.getOperand(l) === res) return true;
      }
    }
    return false;
  }
}
