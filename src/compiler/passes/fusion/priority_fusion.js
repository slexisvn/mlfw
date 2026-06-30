import { FunctionPass, PassResult } from '../pass.js';
import { registry } from '../../ir/graph/ops.js';
import { TensorType, DYNAMIC } from '../../ir/graph/types.js';
import { FusionKind, FusionLegality } from './fusion_analysis.js';
import { FusionGroup } from './fusion_groups.js';
import { FusionCostModel } from './fusion_cost.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { GraphCycles } from './graph_cycles.js';
import { MaxHeap } from './binary_heap.js';
import { materializeFusionGroup } from './fusion_utils.js';
import { canInlineFuse } from '../lowering/graph_to_tensor.js';
import { TraceLevel } from '../../pipeline/trace.js';

function isFusibleOp(op) {
  const def = registry.get(op.opName);
  if (!def) return false;
  if (def.isConstant || def.isTerminator || def.isOpaque) return false;
  if (def.isReduction) return true;
  return canInlineFuse(op.opName);
}

export class PriorityFusionPass extends FunctionPass {
  constructor(config = {}) {
    super('PriorityFusionPass');
    this.requiredAnalyses = [UseDefAnalysis];
    const target = config.target || {};
    this.maxReductions = config.maxReductions ?? 1;
    this.legality = new FusionLegality({
      maxFusionSize: target.maxFusionSize || config.maxFusionSize,
      maxSharedMemory: target.sharedMemoryBytes || config.maxSharedMemory,
      libraryOps: target.libraryOps || config.libraryOps,
      allowReductionFusion: config.allowReductionFusion,
    });
    const benefitWeights = (target.getAttr && target.getAttr('fusionBenefitWeights')) || config.benefitWeights;
    this.costModel = new FusionCostModel({
      memoryBandwidthGBs: target.memoryBandwidthGBs,
      computeTFLOPs: target.computeTFLOPs,
      maxRegistersPerThread: target.registersPerThread,
      maxSharedMemory: target.sharedMemoryBytes,
      libraryOps: target.libraryOps,
      policy: target.getAttr ? target.getAttr('fusionPolicy') : null,
      benefitWeights,
      ...config.cost,
    });
  }

  run(func, analysisManager) {
    const useDef = analysisManager
      ? analysisManager.getAnalysis(UseDefAnalysis, func)
      : UseDefAnalysis.compute(func);
    const topo = useDef.topologicalOrder;
    const n = topo.length;
    if (n === 0) return PassResult.UNCHANGED;

    const idOf = new Map();
    for (let i = 0; i < n; i++) idOf.set(topo[i], i);

    const edges = [];
    for (let i = 0; i < n; i++) {
      const op = topo[i];
      for (let k = 0; k < op.numOperands; k++) {
        const p = op.getOperand(k).definingOp;
        if (p && idOf.has(p)) edges.push([idOf.get(p), i]);
      }
    }

    const cycles = new GraphCycles(n, edges);
    const version = new Int32Array(n);
    const groupOf = new Map();
    for (let i = 0; i < n; i++) {
      if (isFusibleOp(topo[i])) {
        const g = new FusionGroup(i);
        g.addOp(topo[i]);
        groupOf.set(i, g);
      }
    }

    const heap = new MaxHeap();

    const mergedGroup = (ga, gb) => {
      const g = new FusionGroup(-1);
      for (const op of ga.ops) g.addOp(op);
      for (const op of gb.ops) g.addOp(op);
      return g;
    };

    const legalMerge = (merged) => {
      if (merged.size < 2) return false;
      if (!merged.allOpsInlineFusable()) return false;
      let reductions = 0;
      for (const op of merged.ops) {
        const def = registry.get(op.opName);
        if (def && def.isReduction) reductions++;
      }
      if (reductions > this.maxReductions) return false;
      merged.classifyKind();
      if (merged.kind === FusionKind.OPAQUE) return false;
      return this.costModel.shouldFuse(merged).fuse === true;
    };

    const edgeBytes = (prodOp, consOp) => {
      let bytes = 0;
      for (let i = 0; i < consOp.numOperands; i++) {
        const v = consOp.getOperand(i);
        if (v.definingOp === prodOp && v.type instanceof TensorType) {
          const b = v.type.sizeInBytes();
          if (b !== DYNAMIC) bytes += b;
        }
      }
      return bytes;
    };

    const pushCandidate = (prodOp, consOp) => {
      const a = cycles.find(idOf.get(prodOp));
      const b = cycles.find(idOf.get(consOp));
      if (a === b || !groupOf.has(a) || !groupOf.has(b)) return;
      if (!this.legality.canFuse(prodOp, consOp).legal) return;
      const benefit = this.costModel.edgeBenefit(edgeBytes(prodOp, consOp));
      heap.push(benefit, { a, b, va: version[a], vb: version[b], prodOp, consOp });
    };

    const reEval = (group, root) => {
      const dedup = new Set();
      for (const op of group.ops) {
        for (let r = 0; r < op.numResults; r++) {
          for (const use of op.getResult(r).uses()) {
            const uid = idOf.get(use.user);
            if (uid === undefined) continue;
            const ru = cycles.find(uid);
            if (ru === root || !groupOf.has(ru)) continue;
            const key = root * n + ru;
            if (dedup.has(key)) continue;
            dedup.add(key);
            pushCandidate(op, use.user);
          }
        }
        for (let k = 0; k < op.numOperands; k++) {
          const p = op.getOperand(k).definingOp;
          if (!p) continue;
          const pid = idOf.get(p);
          if (pid === undefined) continue;
          const rp = cycles.find(pid);
          if (rp === root || !groupOf.has(rp)) continue;
          const key = rp * n + root;
          if (dedup.has(key)) continue;
          dedup.add(key);
          pushCandidate(p, op);
        }
      }
    };

    const seeded = new Set();
    for (let i = 0; i < n; i++) {
      if (!groupOf.has(i)) continue;
      const op = topo[i];
      for (let k = 0; k < op.numOperands; k++) {
        const p = op.getOperand(k).definingOp;
        if (!p) continue;
        const pid = idOf.get(p);
        if (pid === undefined || !groupOf.has(pid)) continue;
        const key = pid * n + i;
        if (seeded.has(key)) continue;
        seeded.add(key);
        pushCandidate(p, op);
      }
    }

    let mergeCount = 0;
    while (!heap.isEmpty()) {
      const cand = heap.pop();
      const ra = cycles.find(cand.a);
      const rb = cycles.find(cand.b);
      if (ra === rb) continue;
      if (version[ra] !== cand.va || version[rb] !== cand.vb) continue;
      const ga = groupOf.get(ra);
      const gb = groupOf.get(rb);
      if (!ga || !gb) continue;
      if (!this.legality.canFuse(cand.prodOp, cand.consOp).legal) continue;
      const merged = mergedGroup(ga, gb);
      if (!legalMerge(merged)) continue;
      if (cycles.wouldCreateCycle(ra, rb)) continue;

      const s = cycles.merge(ra, rb);
      const big = ga.size >= gb.size ? ga : gb;
      const small = big === ga ? gb : ga;
      big.merge(small);
      groupOf.delete(ra);
      groupOf.delete(rb);
      groupOf.set(s, big);
      version[s] = (version[ra] > version[rb] ? version[ra] : version[rb]) + 1;
      mergeCount++;
      reEval(big, s);
    }

    if (mergeCount === 0) return PassResult.UNCHANGED;

    const finalGroups = [];
    const seenGroups = new Set();
    for (const g of groupOf.values()) {
      if (seenGroups.has(g) || g.size < 2) continue;
      seenGroups.add(g);
      finalGroups.push(g);
    }

    let materialized = 0;
    for (const g of finalGroups) {
      g.classifyKind();
      this._explain(g);
      if (materializeFusionGroup(g, FusionKind.ELEMENTWISE)) materialized++;
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        ops: n, merges: mergeCount, fusions: materialized,
        level: TraceLevel.DEBUG,
      });
    }

    return materialized > 0 ? PassResult.CHANGED : PassResult.UNCHANGED;
  }

  _explain(group) {
    if (!this.trace || !this.trace.explainsEnabled) return;
    const ops = group.ops.map(o => o.opName);
    this.trace.explain('fusion', ops.join('+'), 'fused', null, { groupSize: ops.length, strategy: 'priority' });
  }
}

export { PassResult };
