import { FunctionPass, PassResult } from '../pass.js';
import { registry } from '../../ir/graph/ops.js';
import { FusionKind, classifyOpPattern, canFusePatterns } from './fusion_analysis.js';
import { FusionGroup } from './fusion_groups.js';
import { FusionCostModel } from './fusion_cost.js';
import { PostDominanceAnalysis } from '../../analysis/dominance.js';
import { TraceLevel } from '../../pipeline/trace.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { materializeFusionGroup } from './fusion_utils.js';
import { isConstantOp, isTerminatorOp } from '../../ir/graph/op_traits.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { FusionCostConfig } from './fusion_cost.js';
import type { FusionAwareTarget } from '../../pipeline/pipeline_types.js';

type PostDominance = ReturnType<typeof PostDominanceAnalysis.compute>;

export type DominatorFusionConfig = {
  target?: Partial<FusionAwareTarget> | null;
  maxFusionSize?: number;
  maxReductions?: number;
  hasLibraryOp?: (opName: string) => boolean;
  cost?: FusionCostConfig;
};

function isSkipOp(opName: string): boolean {
  return isTerminatorOp(opName) || isConstantOp(opName);
}

export class DominatorFusionPass extends FunctionPass {
  maxFusionSize: number;
  maxReductions: number;
  hasLibraryOp: (opName: string) => boolean;
  costModel: FusionCostModel;

  constructor(config: DominatorFusionConfig = {}) {
    super('DominatorFusionPass');
    this.requiredAnalyses = [UseDefAnalysis, PostDominanceAnalysis];
    const target = config.target || {};
    this.maxFusionSize = target.maxFusionSize ?? config.maxFusionSize ?? 512;
    this.maxReductions = config.maxReductions || 1;
    this.hasLibraryOp = target.hasLibraryOp ? (opName: string) => (target.hasLibraryOp as (n: string) => boolean)(opName) : (config.hasLibraryOp || (() => false));
    this.costModel = new FusionCostModel({
      memoryBandwidthGBs: target.memoryBandwidthGBs,
      computeTFLOPs: target.computeTFLOPs,
      maxRegistersPerThread: target.registersPerThread,
      maxSharedMemory: target.sharedMemoryBytes,
      hasLibraryOp: target.hasLibraryOp ? (opName: string) => (target.hasLibraryOp as (n: string) => boolean)(opName) : undefined,
      ...config.cost,
    });
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    const useDef = this.getAnalysis(UseDefAnalysis, graphFunc, analysisManager);

    const pdom = this.getAnalysis(PostDominanceAnalysis, graphFunc, analysisManager);
    const topo = useDef.topologicalOrder;
    const groups = this._buildGroups(topo, pdom);

    if (groups.length === 0) return PassResult.UNCHANGED;

    const filtered: FusionGroup[] = [];
    for (const group of groups) {
      if (!group.allOpsInlineFusable()) {
        this._explain(group, false, 'group contains ops without inline fusion support');
        continue;
      }
      const decision = this.costModel.shouldFuse(group);
      this._explain(group, decision.fuse, decision.reason);
      if (decision.fuse) filtered.push(group);
    }

    if (filtered.length === 0) return PassResult.UNCHANGED;

    for (const group of filtered) {
      materializeFusionGroup(group, FusionKind.ELEMENTWISE);
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

  _explain(group: FusionGroup, fuse: boolean, reason: string | null): void {
    if (!this.trace || !this.trace.explainsEnabled) return;
    const ops = group.ops.map(o => o.opName);
    this.trace.explain('fusion', ops.join('+'), fuse ? 'fused' : 'not-fused', reason || null, { groupSize: ops.length, strategy: 'dominator' });
  }

  _buildGroups(topo: readonly Operation[], pdom: PostDominance): FusionGroup[] {
    const opToGroup = new Map<Operation, FusionGroup>();
    const groupList: FusionGroup[] = [];
    let nextId = 0;

    for (let i = topo.length - 1; i >= 0; i--) {
      const op = topo[i];
      if (isSkipOp(op.opName)) continue;

      const pattern = classifyOpPattern(op);
      if (pattern === FusionKind.OPAQUE) continue;
      if (this.hasLibraryOp(op.opName)) continue;

      const pdomOp = pdom.immediatePDom(op) as Operation | null;
      if (!pdomOp || isSkipOp(pdomOp.opName)) continue;

      const pdomPattern = classifyOpPattern(pdomOp);
      if (pdomPattern === FusionKind.OPAQUE) continue;
      if (this.hasLibraryOp(pdomOp.opName)) continue;

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
          if (!this._checkGroupReductions(group)) continue;
          opToGroup.set(op, group);
          opToGroup.set(pdomOp, group);
          groupList.push(group);
          this._absorbIntermediates(op, pdomOp, group, opToGroup, topo, pdom);
        }
      }
    }

    const seen = new Set<FusionGroup>();
    const result: FusionGroup[] = [];
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

    const accepted: FusionGroup[] = [];
    for (const group of result) {
      if (!this._checkGroupReductions(group)) {
        this._explain(group, false, `group exceeds the ${this.maxReductions}-reduction limit`);
        continue;
      }
      if (this._createsCycle(group)) {
        this._explain(group, false, 'fusing would create a dependency cycle');
        continue;
      }
      accepted.push(group);
    }
    return accepted;
  }

  _createsCycle(group: FusionGroup): boolean {
    const inputDefs = new Set<Operation>();
    for (const v of group.getInputValues()) {
      if (v.definingOp && !group.hasOp(v.definingOp)) inputDefs.add(v.definingOp);
    }
    if (inputDefs.size === 0) return false;

    const visited = new Set<Operation>();
    const stack: Operation[] = [];
    for (const v of group.getOutputValues()) {
      for (const use of v.uses()) {
        if (!group.hasOp(use.user)) stack.push(use.user);
      }
    }

    while (stack.length > 0) {
      const op = stack.pop() as Operation;
      if (visited.has(op)) continue;
      visited.add(op);
      if (inputDefs.has(op)) return true;
      for (let r = 0; r < op.numResults; r++) {
        for (const use of op.getResult(r).uses()) {
          if (!group.hasOp(use.user)) stack.push(use.user);
        }
      }
    }
    return false;
  }

  _canFusePatterns(producer: string, consumer: string): boolean {
    return canFusePatterns(producer, consumer);
  }

  _pathAllFusable(from: Operation, to: Operation, topo: readonly Operation[], pdom: PostDominance): boolean {
    const visited = new Set<Operation>();
    const worklist: Operation[] = [from];
    visited.add(from);
    visited.add(to);

    while (worklist.length > 0) {
      const cur = worklist.pop() as Operation;
      for (let r = 0; r < cur.numResults; r++) {
        for (const use of cur.getResult(r).uses()) {
          const user = use.user;
          if (user === to) continue;
          if (visited.has(user)) continue;
          if (!pdom.postDominates(to, user)) continue;
          visited.add(user);

          const pattern = classifyOpPattern(user);
          if (pattern === FusionKind.OPAQUE) return false;
          if (this.hasLibraryOp(user.opName)) return false;

          worklist.push(user);
        }
      }
    }
    return true;
  }

  _absorbIntermediates(from: Operation, to: Operation, group: FusionGroup, opToGroup: Map<Operation, FusionGroup>, topo: readonly Operation[], pdom: PostDominance): void {
    const visited = new Set<Operation>();
    const worklist: Operation[] = [from];
    visited.add(from);
    visited.add(to);

    while (worklist.length > 0) {
      const cur = worklist.pop() as Operation;
      for (let r = 0; r < cur.numResults; r++) {
        for (const use of cur.getResult(r).uses()) {
          const user = use.user;
          if (user === to || visited.has(user)) continue;
          if (!pdom.postDominates(to, user)) continue;
          visited.add(user);

          const pattern = classifyOpPattern(user);
          if (pattern === FusionKind.OPAQUE || this.hasLibraryOp(user.opName)) continue;

          if (group.size < this.maxFusionSize && this._checkReductionLimit(group, user)) {
            group.addOp(user);
            opToGroup.set(user, group);
          }
          worklist.push(user);
        }
      }
    }
  }

  _checkGroupReductions(group: FusionGroup): boolean {
    let count = 0;
    for (const op of group.ops) {
      const def = registry.get(op.opName);
      if (def && def.isReduction) count++;
    }
    return count <= this.maxReductions;
  }

  _checkReductionLimit(group: FusionGroup, newOp: Operation): boolean {
    const newDef = registry.get(newOp.opName);
    if (!newDef || !newDef.isReduction) return true;

    let count = 0;
    for (const op of group.ops) {
      const def = registry.get(op.opName);
      if (def && def.isReduction) count++;
    }
    return count < this.maxReductions;
  }

}
