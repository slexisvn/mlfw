import { TensorType } from '../../ir/graph/types.js';
import { registry } from '../../ir/graph/ops.js';
import { classifyFusionKind, FusionKind } from './fusion_analysis.js';
import { canInlineFuse } from '../lowering/graph_to_tensor.js';
import { OpGroup } from '../partition/op_group.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { OpDef } from '../../ir/graph/op_registry.js';
import type { FusionLegality, FusionLegalityResult } from './fusion_analysis.js';

export class FusionGroup extends OpGroup {
  kind: string | null;
  minTopoPos: number;
  maxTopoPos: number;

  constructor(id: number) {
    super(id);
    this.kind = null;
    this.minTopoPos = Infinity;
    this.maxTopoPos = -Infinity;
  }

  addOp(op: Operation, topoPos?: number): boolean {
    if (!super.addOp(op)) return false;
    if (topoPos !== undefined) {
      if (topoPos < this.minTopoPos) this.minTopoPos = topoPos;
      if (topoPos > this.maxTopoPos) this.maxTopoPos = topoPos;
    }
    return true;
  }

  merge(other: FusionGroup): void {
    for (const op of other.ops) {
      this.addOp(op);
    }
    if (other.minTopoPos < this.minTopoPos) this.minTopoPos = other.minTopoPos;
    if (other.maxTopoPos > this.maxTopoPos) this.maxTopoPos = other.maxTopoPos;
  }

  classifyKind(): string | null {
    this.kind = classifyFusionKind(this.ops);
    return this.kind;
  }

  allOpsInlineFusable(): boolean {
    for (const op of this.ops) {
      const def = registry.get(op.opName);
      if (!def) return false;
      if (def.isReduction || def.isConstant) continue;
      if (!canInlineFuse(op.opName)) return false;
    }
    return true;
  }
}

function outputShapeKey(op: Operation): string | null {
  for (let i = 0; i < op.numResults; i++) {
    const t = op.getResult(i).type;
    if (t instanceof TensorType) return t.shape.join(',');
  }
  return null;
}

export class FusionGroupBuilder {
  legality: FusionLegality;
  private _nextId: number;
  private _topoIndex: Map<Operation, number>;

  constructor(legality: FusionLegality) {
    this.legality = legality;
    this._nextId = 0;
    this._topoIndex = new Map();
  }

  buildProducerConsumerGroups(func: GraphFunction): FusionGroup[] {
    this._topoIndex = new Map();
    let idx = 0;
    for (const op of func.ops()) this._topoIndex.set(op, idx++);

    const groups: FusionGroup[] = [];
    const opToGroup = new Map<Operation, FusionGroup>();

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
          const mergeResult: FusionLegalityResult = this.legality.canMergeGroups(consumerGroup as FusionGroup, producerGroup as FusionGroup);
          if (mergeResult.legal && !this._mergeWouldCreateCycle(consumerGroup as FusionGroup, producerGroup as FusionGroup)) {
            (consumerGroup as FusionGroup).merge(producerGroup as FusionGroup);
            for (const pOp of (producerGroup as FusionGroup).ops) {
              opToGroup.set(pOp, consumerGroup as FusionGroup);
            }
          }
        }
      }
    }

    const seen = new Set<FusionGroup>();
    for (const group of opToGroup.values()) {
      if (seen.has(group) || group.size < 2) continue;
      seen.add(group);
      group.classifyKind();
      groups.push(group);
    }
    return groups;
  }

  _wouldCreateCycle(group: FusionGroup, newOp: Operation): boolean {
    for (let i = 0; i < newOp.numOperands; i++) {
      const producer = newOp.getOperand(i).definingOp;
      if (!producer || group.hasOp(producer)) continue;
      const pos = this._topoIndex.get(producer) as number;
      if (pos < group.minTopoPos) continue;
      if (this._dependsOnGroup(producer, group)) return true;
    }
    return false;
  }

  _dependsOnOps(startOp: Operation, targetSet: ReadonlySet<Operation>, minPos: number): boolean {
    const visited = new Set<Operation>();
    const worklist: Operation[] = [startOp];
    visited.add(startOp);
    while (worklist.length > 0) {
      const op = worklist.pop() as Operation;
      for (let i = 0; i < op.numOperands; i++) {
        const dep = op.getOperand(i).definingOp;
        if (!dep || visited.has(dep)) continue;
        if (targetSet.has(dep)) return true;
        const depPos = this._topoIndex.get(dep) as number;
        if (depPos < minPos) continue;
        visited.add(dep);
        worklist.push(dep);
      }
    }
    return false;
  }

  _dependsOnGroup(startOp: Operation, group: FusionGroup): boolean {
    return this._dependsOnOps(startOp, group.opSet, group.minTopoPos);
  }

  _mergeWouldCreateCycle(groupA: FusionGroup, groupB: FusionGroup): boolean {
    const minPos = Math.min(groupA.minTopoPos, groupB.minTopoPos);
    const maxPos = Math.max(groupA.maxTopoPos, groupB.maxTopoPos);
    const mergedSet = new Set([...groupA.opSet, ...groupB.opSet]);
    for (const op of mergedSet) {
      for (let i = 0; i < op.numOperands; i++) {
        const producer = op.getOperand(i).definingOp;
        if (!producer || mergedSet.has(producer)) continue;
        const pos = this._topoIndex.get(producer) as number;
        if (pos < minPos || pos > maxPos) continue;
        if (this._dependsOnOps(producer, mergedSet, minPos)) return true;
      }
    }
    return false;
  }

  _bucketable(op: Operation, def: OpDef | null | undefined): boolean {
    return !!def && !def.isConstant && !def.isTerminator && !def.isOpaque
      && op.numResults > 0 && outputShapeKey(op) !== null;
  }

  buildHorizontalGroups(func: GraphFunction): FusionGroup[] {
    const topoOps = [...func.ops()];
    this._topoIndex = new Map();
    for (let i = 0; i < topoOps.length; i++) this._topoIndex.set(topoOps[i], i);

    const groups: FusionGroup[] = [];
    const opToGroup = new Map<Operation, FusionGroup>();
    const window = this.legality.maxFusionSize || topoOps.length;
    const taint = new Map<Operation, number>();
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
        if (!(this._sharesInput(op1, op) || ((def1 as OpDef).isElementwise && (def as OpDef).isElementwise))) continue;

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

  buildAllGroups(func: GraphFunction): FusionGroup[] {
    const pcGroups = this.buildProducerConsumerGroups(func);
    const horizontalGroups = this.buildHorizontalGroups(func);

    const opToRep = new Map<Operation, Operation | FusionGroup>();
    for (const g of pcGroups) {
      for (const op of g.ops) opToRep.set(op, g);
    }

    const fusedOps = new Set(opToRep.keys());
    const candidates: FusionGroup[] = [];
    for (const h of horizontalGroups) {
      let overlaps = false;
      for (const op of h.ops) {
        if (fusedOps.has(op)) { overlaps = true; break; }
      }
      if (overlaps) continue;
      candidates.push(h);
    }

    const allOps = [...func.ops()];
    const opEdges: Operation[] = [];
    for (const op of allOps) {
      for (let i = 0; i < op.numOperands; i++) {
        const def = op.getOperand(i).definingOp;
        if (def) opEdges.push(op, def);
      }
    }

    for (const h of candidates) {
      for (const op of h.ops) opToRep.set(op, h);
    }
    if (!this._condensedHasCycle(allOps, opEdges, opToRep)) {
      return [...pcGroups, ...candidates];
    }

    for (const h of candidates) {
      for (const op of h.ops) opToRep.delete(op);
    }
    const result = [...pcGroups];
    for (const h of candidates) {
      for (const op of h.ops) opToRep.set(op, h);
      if (this._condensedHasCycle(allOps, opEdges, opToRep)) {
        for (const op of h.ops) opToRep.delete(op);
        continue;
      }
      result.push(h);
    }

    return result;
  }

  _condensedHasCycle(allOps: readonly Operation[], opEdges: readonly Operation[], opToRep: ReadonlyMap<Operation, Operation | FusionGroup>): boolean {
    type CondensedNode = Operation | FusionGroup;
    const repOf = (op: Operation): CondensedNode => opToRep.get(op) || op;
    const adj = new Map<CondensedNode, Set<CondensedNode>>();
    const nodes = new Set<CondensedNode>();
    for (const op of allOps) nodes.add(repOf(op));
    for (let e = 0; e < opEdges.length; e += 2) {
      const r = repOf(opEdges[e]);
      const pr = repOf(opEdges[e + 1]);
      if (pr === r) continue;
      nodes.add(pr);
      nodes.add(r);
      let succ = adj.get(pr);
      if (!succ) { succ = new Set(); adj.set(pr, succ); }
      succ.add(r);
    }

    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map<CondensedNode, number>();
    for (const n of nodes) color.set(n, WHITE);
    for (const start of nodes) {
      if (color.get(start) !== WHITE) continue;
      const stack: CondensedNode[] = [start];
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

  _sharesInput(op1: Operation, op2: Operation): boolean {
    if (op1.numOperands === 0 || op2.numOperands === 0) return false;
    const op1Operands = new Set<Value>();
    for (let k = 0; k < op1.numOperands; k++) {
      op1Operands.add(op1.getOperand(k));
    }
    for (let l = 0; l < op2.numOperands; l++) {
      if (op1Operands.has(op2.getOperand(l))) return true;
    }
    return false;
  }

}
