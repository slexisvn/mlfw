import { FunctionPass, PassResult } from '../pass.js';
import { Operation, cloneRegion } from '../../ir/graph/operation.js';
import { Block, Region } from '../../ir/graph/block.js';
import { registry } from '../../ir/graph/ops.js';
import { FusionKind, classifyFusionKind, classifyOpPattern } from './fusion_analysis.js';
import { FusionGroup } from './fusion_groups.js';
import { FusionCostModel } from './fusion_cost.js';
import { PostDominanceAnalysis } from '../../analysis/dominance.js';
import { TraceLevel } from '../../pipeline/trace.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';

const SKIP_OPS = new Set(['return', 'yield', 'constant', 'scalar_constant']);

export class DominatorFusionPass extends FunctionPass {
  constructor(config = {}) {
    super('DominatorFusionPass');
    const target = config.target || {};
    this.maxFusionSize = target.maxFusionSize || config.maxFusionSize || 512;
    this.maxReductions = config.maxReductions || 1;
    this.libraryOps = target.libraryOps || config.libraryOps || new Set();
    this.costModel = new FusionCostModel({
      memoryBandwidthGBs: target.memoryBandwidthGBs,
      computeTFLOPs: target.computeTFLOPs,
      maxRegistersPerThread: target.registersPerThread,
      maxSharedMemory: target.sharedMemoryBytes,
      libraryOps: target.libraryOps,
      ...config.cost,
    });
  }

  run(func, analysisManager) {
    const useDef = analysisManager
      ? analysisManager.getAnalysis(UseDefAnalysis, func)
      : UseDefAnalysis.compute(func);

    const pdom = PostDominanceAnalysis.compute(func, { useDef });
    const topo = useDef.topologicalOrder;
    const groups = this._buildGroups(topo, pdom);

    if (groups.length === 0) return PassResult.UNCHANGED;

    const filtered = [];
    for (const group of groups) {
      if (!group.allOpsInlineFusable()) continue;
      const decision = this.costModel.shouldFuse(group);
      if (decision.fuse) filtered.push(group);
    }

    if (filtered.length === 0) return PassResult.UNCHANGED;

    for (const group of filtered) {
      this._materialize(func, group);
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        groupsBuilt: groups.length, groupsFused: filtered.length,
        level: TraceLevel.DEBUG,
      });
    }

    return PassResult.CHANGED;
  }

  _buildGroups(topo, pdom) {
    const opToGroup = new Map();
    const groupList = [];
    let nextId = 0;

    for (let i = topo.length - 1; i >= 0; i--) {
      const op = topo[i];
      if (SKIP_OPS.has(op.opName)) continue;

      const pattern = classifyOpPattern(op);
      if (pattern === FusionKind.OPAQUE) continue;
      if (this.libraryOps.has(op.opName)) continue;

      const pdomOp = pdom.immediatePDom(op);
      if (!pdomOp || SKIP_OPS.has(pdomOp.opName)) continue;

      const pdomPattern = classifyOpPattern(pdomOp);
      if (pdomPattern === FusionKind.OPAQUE) continue;
      if (this.libraryOps.has(pdomOp.opName)) continue;

      if (!this._canFusePatterns(pattern, pdomPattern)) continue;
      if (!this._pathAllFusable(op, pdomOp, topo, pdom)) continue;

      const existingGroup = opToGroup.get(pdomOp);
      if (existingGroup) {
        if (existingGroup.size + 1 > this.maxFusionSize) continue;
        if (!this._checkReductionLimit(existingGroup, op)) continue;
        existingGroup.addOp(op);
        opToGroup.set(op, existingGroup);
        this._absorbIntermediates(op, pdomOp, existingGroup, opToGroup, topo, pdom);
      } else {
        const myGroup = opToGroup.get(op);
        if (myGroup) {
          if (myGroup.size + 1 > this.maxFusionSize) continue;
          if (!this._checkReductionLimit(myGroup, pdomOp)) continue;
          myGroup.addOp(pdomOp);
          opToGroup.set(pdomOp, myGroup);
          this._absorbIntermediates(op, pdomOp, myGroup, opToGroup, topo, pdom);
        } else {
          const group = new FusionGroup(nextId++);
          group.addOp(op);
          group.addOp(pdomOp);
          opToGroup.set(op, group);
          opToGroup.set(pdomOp, group);
          groupList.push(group);
          this._absorbIntermediates(op, pdomOp, group, opToGroup, topo, pdom);
        }
      }
    }

    const seen = new Set();
    const result = [];
    for (const group of groupList) {
      if (seen.has(group) || group.size < 2) continue;
      seen.add(group);
      group.classifyKind();
      result.push(group);
    }
    for (const group of opToGroup.values()) {
      if (seen.has(group) || group.size < 2) continue;
      seen.add(group);
      group.classifyKind();
      result.push(group);
    }

    return result;
  }

  _canFusePatterns(producer, consumer) {
    if (producer === FusionKind.ELEMENTWISE && consumer === FusionKind.ELEMENTWISE) return true;
    if (producer === FusionKind.BROADCAST && consumer === FusionKind.ELEMENTWISE) return true;
    if (producer === FusionKind.INJECTIVE && consumer === FusionKind.ELEMENTWISE) return true;
    if (producer === FusionKind.ELEMENTWISE && consumer === FusionKind.INJECTIVE) return true;
    if (producer === FusionKind.ELEMENTWISE && consumer === FusionKind.REDUCTION) return true;
    if (producer === FusionKind.BROADCAST && consumer === FusionKind.REDUCTION) return true;
    if (producer === FusionKind.INJECTIVE && consumer === FusionKind.REDUCTION) return true;
    if (producer === FusionKind.REDUCTION && consumer === FusionKind.ELEMENTWISE) return true;
    if (producer === FusionKind.BROADCAST && consumer === FusionKind.INJECTIVE) return true;
    if (producer === FusionKind.INJECTIVE && consumer === FusionKind.INJECTIVE) return true;
    return false;
  }

  _pathAllFusable(from, to, topo, pdom) {
    const visited = new Set();
    const worklist = [from];
    visited.add(from);
    visited.add(to);

    while (worklist.length > 0) {
      const cur = worklist.pop();
      for (let r = 0; r < cur.numResults; r++) {
        for (const use of cur.getResult(r).uses()) {
          const user = use.user;
          if (user === to) continue;
          if (visited.has(user)) continue;
          if (!pdom.postDominates(to, user)) continue;
          visited.add(user);

          const pattern = classifyOpPattern(user);
          if (pattern === FusionKind.OPAQUE) return false;
          if (this.libraryOps.has(user.opName)) return false;

          worklist.push(user);
        }
      }
    }
    return true;
  }

  _absorbIntermediates(from, to, group, opToGroup, topo, pdom) {
    const visited = new Set();
    const worklist = [from];
    visited.add(from);
    visited.add(to);

    while (worklist.length > 0) {
      const cur = worklist.pop();
      for (let r = 0; r < cur.numResults; r++) {
        for (const use of cur.getResult(r).uses()) {
          const user = use.user;
          if (user === to || visited.has(user)) continue;
          if (!pdom.postDominates(to, user)) continue;
          visited.add(user);

          const pattern = classifyOpPattern(user);
          if (pattern === FusionKind.OPAQUE || this.libraryOps.has(user.opName)) continue;

          if (group.size < this.maxFusionSize && this._checkReductionLimit(group, user)) {
            group.addOp(user);
            opToGroup.set(user, group);
          }
          worklist.push(user);
        }
      }
    }
  }

  _checkReductionLimit(group, newOp) {
    const newDef = registry.get(newOp.opName);
    if (!newDef || !newDef.isReduction) return true;

    let count = 0;
    for (const op of group.ops) {
      const def = registry.get(op.opName);
      if (def && def.isReduction) count++;
    }
    return count < this.maxReductions;
  }

  _materialize(func, group) {
    const sorted = this._topoSort(group);
    if (!sorted || sorted.length === 0) return;

    const inputValues = group.getInputValues();
    const outputValues = group.getOutputValues();

    const inputTypes = inputValues.map(v => v.type);
    const outputTypes = outputValues.map(v => v.type);

    const bodyRegion = new Region();
    const bodyBlock = new Block(inputTypes);
    bodyRegion.addBlock(bodyBlock);

    const valueMap = new Map();
    for (let i = 0; i < inputValues.length; i++) {
      valueMap.set(inputValues[i], bodyBlock.arguments[i]);
    }

    for (const op of sorted) {
      const mappedOperands = [];
      for (let i = 0; i < op.numOperands; i++) {
        const orig = op.getOperand(i);
        const mapped = valueMap.get(orig);
        mappedOperands.push(mapped !== undefined ? mapped : orig);
      }
      const resultTypes = op.results.map(r => r.type);
      const clonedRegions = op.regions.length > 0 ? op.regions.map(r => cloneRegion(r)) : null;
      const cloned = new Operation(
        op.opName, mappedOperands, resultTypes,
        new Map(op.attributes), clonedRegions
      );
      bodyBlock.pushOp(cloned);
      for (let i = 0; i < op.numResults; i++) {
        valueMap.set(op.getResult(i), cloned.getResult(i));
      }
    }

    const yieldValues = outputValues.map(v => {
      const mapped = valueMap.get(v);
      if (mapped === undefined) throw new Error('DominatorFusion: output not found in valueMap');
      return mapped;
    });
    bodyBlock.pushOp(new Operation('yield', yieldValues, []));

    const fusionOp = new Operation(
      'fusion', inputValues, outputTypes,
      { fusion_kind: group.kind || FusionKind.ELEMENTWISE },
      [bodyRegion]
    );

    const firstOp = sorted[0];
    if (!firstOp.parentBlock) return;
    firstOp.parentBlock.insertBefore(fusionOp, firstOp);

    for (let i = 0; i < outputValues.length; i++) {
      outputValues[i].replaceAllUsesWith(fusionOp.getResult(i));
    }

    for (const op of sorted) {
      op.dropAllOperands();
      if (op.parentBlock) op.parentBlock.removeOp(op);
    }
  }

  _topoSort(group) {
    const sorted = [];
    const visited = new Set();
    const visiting = new Set();

    const visit = (op) => {
      if (visited.has(op)) return true;
      if (visiting.has(op)) return false;
      visiting.add(op);
      for (let i = 0; i < op.numOperands; i++) {
        const defOp = op.getOperand(i).definingOp;
        if (defOp && group.hasOp(defOp)) {
          if (!visit(defOp)) return false;
        }
      }
      visiting.delete(op);
      visited.add(op);
      sorted.push(op);
      return true;
    };

    for (const op of group.ops) {
      if (!visit(op)) return null;
    }
    return sorted;
  }
}
