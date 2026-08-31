import { FunctionPass, PassResult } from '../pass.js';
import { FusionLegality, FusionKind } from './fusion_analysis.js';
import { FusionGroupBuilder } from './fusion_groups.js';
import { FusionCostModel } from './fusion_cost.js';
import { materializeFusionGroup } from './fusion_utils.js';
import { TraceLevel } from '../../support/trace.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { FusionGroup } from './fusion_groups.js';
import type { FusionCostConfig } from './fusion_cost.js';
import type { FusionAwareTarget } from '../../support/config_types.js';
import { hasLibraryOp } from '../../ir/graph/op_traits.js';
import type { LibraryTarget } from '../../ir/graph/op_traits.js';

export type FusionPassConfig = {
  target?: Partial<FusionAwareTarget> | null;
  maxFusionSize?: number;
  libraryOps?: ReadonlySet<string>;
  allowReductionFusion?: boolean;
  cost?: FusionCostConfig;
};

export class FusionPass extends FunctionPass {
  legality: FusionLegality;
  costModel: FusionCostModel;
  groupBuilder: FusionGroupBuilder;

  constructor(config: FusionPassConfig = {}) {
    super('FusionPass');
    const target = config.target || {};
    this.legality = new FusionLegality({
      maxFusionSize: target.maxFusionSize ?? config.maxFusionSize,
      allowReductionFusion: config.allowReductionFusion,
    });
    this.costModel = new FusionCostModel({
      memoryBandwidthGBs: target.memoryBandwidthGBs,
      computeTFLOPs: target.computeTFLOPs,
      maxRegistersPerThread: target.registersPerThread,
      maxSharedMemory: target.sharedMemoryBytes,
      hasLibraryOp: target.hasLibraryClass ? (opName: string) => hasLibraryOp(target as LibraryTarget, opName) : undefined,
      policy: target.getAttr ? target.getAttr<FusionCostConfig['policy']>('fusionPolicy') ?? null : null,
      ...config.cost,
    });
    this.groupBuilder = new FusionGroupBuilder(this.legality);
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const groups = this.groupBuilder.buildAllGroups(func as GraphFunction);

    const validGroups: FusionGroup[] = [];
    for (const group of groups) {
      if (this._createsCycle(func as GraphFunction, group)) {
        this._traceDecision(group, false, 'fusing would create a dependency cycle');
      } else {
        validGroups.push(group);
      }
    }

    const filteredGroups: FusionGroup[] = [];
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

  _traceDecision(group: FusionGroup, fuse: boolean, reason: string | null): void {
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
    this.trace.explain('fusion', ops.join('+'), fuse ? 'grouped' : 'not-grouped', reason || null, { groupSize: ops.length });
  }

  _createsCycle(func: GraphFunction, group: FusionGroup): boolean {
    const groupSet = group.opSet;
    const outputs = group.getOutputValues();
    const reachable = new Set<Operation>();
    const worklist: Operation[] = [];

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
