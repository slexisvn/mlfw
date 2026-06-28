import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { Block, Region } from '../../ir/graph/block.js';
import { topoSortByOperands } from '../../ir/graph/graph_algorithms.js';
import { FusionLegality, FusionKind } from './fusion_analysis.js';
import { FusionGroupBuilder } from './fusion_groups.js';
import { FusionCostModel } from './fusion_cost.js';
import { makeComesBefore } from './fusion_utils.js';
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
      this._materializeFusion(func, group);
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

  _materializeFusion(func, group) {
    const sortedOps = this._topologicalSort(group);
    if (sortedOps === null || sortedOps.length === 0) return;

    group._inputValues = null;
    group._outputValues = null;
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

    for (const op of sortedOps) {
      bodyBlock.pushOp(op.clone(valueMap));
    }

    const yieldValues = outputValues.map(v => {
      const mapped = valueMap.get(v);
      if (mapped === undefined) {
        throw new Error('Fusion materialization: output value not found in valueMap');
      }
      return mapped;
    });
    const yieldOp = new Operation('yield', yieldValues, []);
    bodyBlock.pushOp(yieldOp);

    const comesBefore = makeComesBefore(sortedOps[0].parentBlock);
    let insertAfter = null;
    for (const val of inputValues) {
      const producer = val.definingOp;
      if (!producer || group.hasOp(producer)) continue;
      if (!insertAfter || !comesBefore(producer, insertAfter)) {
        insertAfter = producer;
      }
    }

    const fusionOp = new Operation(
      'fusion',
      inputValues,
      outputTypes,
      { fusion_kind: group.kind || FusionKind.ELEMENTWISE },
      [bodyRegion]
    );

    const block = sortedOps[0].parentBlock;
    if (!block) return;

    if (insertAfter) {
      block.insertAfter(fusionOp, insertAfter);
    } else {
      block.insertBefore(fusionOp, sortedOps[0]);
    }

    for (let i = 0; i < outputValues.length; i++) {
      outputValues[i].replaceAllUsesWith(fusionOp.getResult(i));
    }

    for (const op of sortedOps) {
      op.dropAllOperands();
      if (op.parentBlock) {
        op.parentBlock.removeOp(op);
      }
    }
  }

  _topologicalSort(group) {
    return topoSortByOperands(group.ops, (op) => group.hasOp(op), 'null');
  }
}

export { PassResult };
