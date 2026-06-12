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
    this.minTopoPos = Infinity;
    this.maxTopoPos = -Infinity;
  }

  addOp(op, topoPos) {
    if (this.opSet.has(op)) return;
    this.ops.push(op);
    this.opSet.add(op);
    this._inputValues = null;
    this._outputValues = null;
    if (topoPos !== undefined) {
      if (topoPos < this.minTopoPos) this.minTopoPos = topoPos;
      if (topoPos > this.maxTopoPos) this.maxTopoPos = topoPos;
    }
  }

  hasOp(op) {
    return this.opSet.has(op);
  }

  merge(other) {
    for (const op of other.ops) {
      this.addOp(op);
    }
    if (other.minTopoPos < this.minTopoPos) this.minTopoPos = other.minTopoPos;
    if (other.maxTopoPos > this.maxTopoPos) this.maxTopoPos = other.maxTopoPos;
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
    this._topoIndex = null;
  }

  buildProducerConsumerGroups(func) {
    this._topoIndex = new Map();
    let idx = 0;
    for (const op of func.ops()) this._topoIndex.set(op, idx++);

    const groups = [];
    const opToGroup = new Map();

    for (const op of func.ops()) {
      const def = registry.get(op.opName);
      if (!def || def.isConstant || def.isTerminator) continue;
      if (def.isReduction) continue;
      const opPos = this._topoIndex.get(op);

      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (!producer) continue;
        const pDef = registry.get(producer.opName);
        if (!pDef || pDef.isConstant) continue;
        if (pDef.isReduction) continue;

        const consumerGroup = opToGroup.get(op);
        const producerGroup = opToGroup.get(producer);
        if (consumerGroup && consumerGroup === producerGroup) continue;

        const result = this.legality.canFuse(producer, op);
        if (!result.legal) continue;

        const producerPos = this._topoIndex.get(producer);
        if (!consumerGroup && !producerGroup) {
          const group = new FusionGroup(this._nextId++);
          group.addOp(producer, producerPos);
          group.addOp(op, opPos);
          opToGroup.set(producer, group);
          opToGroup.set(op, group);
        } else if (consumerGroup && !producerGroup) {
          if (consumerGroup.size < this.legality.maxFusionSize && !this._wouldCreateCycle(consumerGroup, producer)) {
            consumerGroup.addOp(producer, producerPos);
            opToGroup.set(producer, consumerGroup);
          }
        } else if (!consumerGroup && producerGroup) {
          if (producerGroup.size < this.legality.maxFusionSize && !this._wouldCreateCycle(producerGroup, op)) {
            producerGroup.addOp(op, opPos);
            opToGroup.set(op, producerGroup);
          }
        } else {
          const mergeResult = this.legality.canMergeGroups(consumerGroup, producerGroup);
          if (mergeResult.legal && !this._mergeWouldCreateCycle(consumerGroup, producerGroup)) {
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

  _wouldCreateCycle(group, newOp) {
    for (let i = 0; i < newOp.numOperands; i++) {
      const producer = newOp.getOperand(i).definingOp;
      if (!producer || group.hasOp(producer)) continue;
      const pos = this._topoIndex.get(producer);
      if (pos < group.minTopoPos) continue;
      if (this._dependsOnGroup(producer, group)) return true;
    }
    return false;
  }

  _dependsOnOps(startOp, targetSet, minPos) {
    const visited = new Set();
    const worklist = [startOp];
    visited.add(startOp);
    while (worklist.length > 0) {
      const op = worklist.pop();
      for (let i = 0; i < op.numOperands; i++) {
        const dep = op.getOperand(i).definingOp;
        if (!dep || visited.has(dep)) continue;
        if (targetSet.has(dep)) return true;
        const depPos = this._topoIndex.get(dep);
        if (depPos < minPos) continue;
        visited.add(dep);
        worklist.push(dep);
      }
    }
    return false;
  }

  _dependsOnGroup(startOp, group) {
    return this._dependsOnOps(startOp, group.opSet, group.minTopoPos);
  }

  _mergeWouldCreateCycle(groupA, groupB) {
    const minPos = Math.min(groupA.minTopoPos, groupB.minTopoPos);
    const maxPos = Math.max(groupA.maxTopoPos, groupB.maxTopoPos);
    const mergedSet = new Set([...groupA.opSet, ...groupB.opSet]);
    for (const op of mergedSet) {
      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (!producer || mergedSet.has(producer)) continue;
        const pos = this._topoIndex.get(producer);
        if (pos < minPos || pos > maxPos) continue;
        if (this._dependsOnOps(producer, mergedSet, minPos)) return true;
      }
    }
    return false;
  }

  _bucketable(op, def) {
    return def && !def.isConstant && !def.isTerminator && !def.isOpaque
      && op.numResults > 0 && outputShapeKey(op) !== null;
  }

  buildHorizontalGroups(func) {
    const topoOps = [...func.ops()];
    this._topoIndex = new Map();
    for (let i = 0; i < topoOps.length; i++) this._topoIndex.set(topoOps[i], i);

    const groups = [];
    const opToGroup = new Map();
    const window = this.legality.maxFusionSize || topoOps.length;
    const taint = new Map();
    let version = 0;

    for (let p1 = 0; p1 < topoOps.length; p1++) {
      const op1 = topoOps[p1];
      if (opToGroup.has(op1)) continue;
      const def1 = registry.get(op1.opName);
      if (!this._bucketable(op1, def1)) continue;
      const type1 = op1.getResult(0).type;

      version++;
      taint.set(op1, version);
      const group = new FusionGroup(this._nextId++);
      group.addOp(op1, p1);
      opToGroup.set(op1, group);

      const end = Math.min(topoOps.length, p1 + 1 + window);
      for (let gp = p1 + 1; gp < end; gp++) {
        const op = topoOps[gp];

        let dependsOnGroup = false;
        for (let k = 0; k < op.numOperands; k++) {
          const d = op.getOperand(k).definingOp;
          if (d && taint.get(d) === version) { dependsOnGroup = true; break; }
        }
        if (dependsOnGroup) { taint.set(op, version); continue; }

        if (group.size >= window) continue;
        if (opToGroup.has(op)) continue;
        const def = registry.get(op.opName);
        if (!this._bucketable(op, def)) continue;
        if (!type1.equals(op.getResult(0).type)) continue;
        if (!(this._sharesInput(op1, op) || (def1.isElementwise && def.isElementwise))) continue;

        group.addOp(op, gp);
        opToGroup.set(op, group);
        taint.set(op, version);
      }

      if (group.size >= 2) {
        group.kind = FusionKind.HORIZONTAL;
        groups.push(group);
      }
    }

    return groups;
  }

  buildAllGroups(func) {
    const pcGroups = this.buildProducerConsumerGroups(func);
    const horizontalGroups = this.buildHorizontalGroups(func);

    const opToRep = new Map();
    for (const g of pcGroups) {
      for (const op of g.ops) opToRep.set(op, g);
    }

    const fusedOps = new Set(opToRep.keys());
    const candidates = [];
    for (const h of horizontalGroups) {
      let overlaps = false;
      for (const op of h.ops) {
        if (fusedOps.has(op)) { overlaps = true; break; }
      }
      if (overlaps) continue;
      candidates.push(h);
    }

    for (const h of candidates) {
      for (const op of h.ops) opToRep.set(op, h);
    }
    if (!this._condensedHasCycle(func, opToRep)) {
      return [...pcGroups, ...candidates];
    }

    for (const h of candidates) {
      for (const op of h.ops) opToRep.delete(op);
    }
    const result = [...pcGroups];
    for (const h of candidates) {
      for (const op of h.ops) opToRep.set(op, h);
      if (this._condensedHasCycle(func, opToRep)) {
        for (const op of h.ops) opToRep.delete(op);
        continue;
      }
      result.push(h);
    }

    return result;
  }

  _condensedHasCycle(func, opToRep) {
    const repOf = (op) => opToRep.get(op) || op;
    const adj = new Map();
    const nodes = new Set();
    for (const op of func.ops()) {
      const r = repOf(op);
      nodes.add(r);
      for (let i = 0; i < op.numOperands; i++) {
        const def = op.getOperand(i).definingOp;
        if (!def) continue;
        const pr = repOf(def);
        if (pr === r) continue;
        nodes.add(pr);
        let succ = adj.get(pr);
        if (!succ) { succ = new Set(); adj.set(pr, succ); }
        succ.add(r);
      }
    }

    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map();
    for (const n of nodes) color.set(n, WHITE);
    for (const start of nodes) {
      if (color.get(start) !== WHITE) continue;
      const stack = [start];
      while (stack.length > 0) {
        const node = stack[stack.length - 1];
        const c = color.get(node);
        if (c === WHITE) {
          color.set(node, GRAY);
          const succ = adj.get(node);
          if (succ) {
            for (const m of succ) {
              const mc = color.get(m);
              if (mc === GRAY) return true;
              if (mc === WHITE) stack.push(m);
            }
          }
        } else {
          if (c === GRAY) color.set(node, BLACK);
          stack.pop();
        }
      }
    }
    return false;
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
