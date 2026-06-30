import { FunctionPass, PassResult } from '../pass.js';
import { FusionLegality, FusionKind } from './fusion_analysis.js';
import { FusionGroupBuilder } from './fusion_groups.js';
import { FusionCostModel } from './fusion_cost.js';
import { materializeFusionGroup } from './fusion_utils.js';
import { TraceLevel } from '../../pipeline/trace.js';

export class FusionPass extends FunctionPass {
  constructor(config = {}) {
    super('FusionPass');
    const target = config.target || {};
    this.legality = new FusionLegality({
      maxFusionSize: target.maxFusionSize || config.maxFusionSize,
      maxSharedMemory: target.sharedMemoryBytes || config.maxSharedMemory,
      libraryOps: target.libraryOps || config.libraryOps,
      allowReductionFusion: config.allowReductionFusion,
    });
    this.costModel = new FusionCostModel({
      memoryBandwidthGBs: target.memoryBandwidthGBs,
      computeTFLOPs: target.computeTFLOPs,
      maxRegistersPerThread: target.registersPerThread,
      maxSharedMemory: target.sharedMemoryBytes,
      libraryOps: target.libraryOps,
      policy: target.getAttr ? target.getAttr('fusionPolicy') : null,
      ...config.cost,
    });
    this.groupBuilder = new FusionGroupBuilder(this.legality);
  }

  run(func, analysisManager) {
    const groups = this.groupBuilder.buildAllGroups(func);

    const validGroups = [];
    for (const group of groups) {
      if (this._createsCycle(func, group)) {
        this._traceDecision(group, false, 'fusing would create a dependency cycle');
      } else {
        validGroups.push(group);
      }
    }

    const filteredGroups = [];
    for (const group of validGroups) {
      if (!group.allOpsInlineFusable()) {
        this._traceDecision(group, false, 'group contains ops without inline fusion support');
        continue;
      }
      const decision = this.costModel.shouldFuse(group);
      this._traceDecision(group, decision.fuse, decision.reason);
      if (decision.fuse) {
        filteredGroups.push(group);
      }
    }

    if (filteredGroups.length === 0) return PassResult.UNCHANGED;

    for (const group of filteredGroups) {
      materializeFusionGroup(group, FusionKind.ELEMENTWISE);
    }

    return PassResult.CHANGED;
  }

  _traceDecision(group, fuse, reason) {
    if (!this.trace || this.trace.level < TraceLevel.DEBUG) return;
    const ops = group.ops.map(o => o.opName);
    this.trace.emit({
      type: 'fusion_decision',
      passName: this.name,
      groupSize: group.ops.length,
      ops,
      anchor: ops[ops.length - 1] || null,
      fuse,
      reason: reason || null,
      level: TraceLevel.DEBUG,
    });
    this.trace.explain('fusion', ops.join('+'), fuse ? 'fused' : 'not-fused', reason || null, { groupSize: ops.length });
  }

  _createsCycle(func, group) {
    const groupSet = group.opSet;
    const outputs = group.getOutputValues();
    const reachable = new Set();
    const worklist = [];

    for (const val of outputs) {
      for (const use of val.uses()) {
        if (!groupSet.has(use.user)) {
          worklist.push(use.user);
          reachable.add(use.user);
        }
      }
    }

    let head = 0;
    while (head < worklist.length) {
      const op = worklist[head++];
      if (groupSet.has(op)) return true;

      for (let i = 0; i < op.numResults; i++) {
        for (const use of op.getResult(i).uses()) {
          if (!reachable.has(use.user)) {
            reachable.add(use.user);
            worklist.push(use.user);
          }
        }
      }
    }

    return false;
  }
}

export { PassResult };
