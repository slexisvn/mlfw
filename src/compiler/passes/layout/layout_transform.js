import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { TensorType, Layout } from '../../ir/graph/types.js';
import { LayoutPolicy } from './layout_policy.js';
import { LayoutAnalysis } from '../../analysis/layout_analysis.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { TraceLevel } from '../../pipeline/trace.js';

export class LayoutTransformPass extends FunctionPass {
  constructor(config = {}) {
    super('LayoutTransformPass');
    this.target = config.target || null;
    this._policy = null;
  }

  run(func, analysisManager) {
    if (!this.target) return PassResult.UNCHANGED;

    if (!this._policy) {
      this._policy = new LayoutPolicy(this.target);
    }

    const useDef = analysisManager
      ? analysisManager.getAnalysis(UseDefAnalysis, func)
      : UseDefAnalysis.compute(func);

    const result = LayoutAnalysis.compute(func, { useDef }, this._policy);

    if (result.conversions.length === 0) return PassResult.UNCHANGED;

    const groups = new Map();
    for (const conv of result.conversions) {
      const { value, consumer, operandIdx, from, to } = conv;
      const key = valueLayoutKey(value, from, to);
      let g = groups.get(key);
      if (!g) {
        g = { value, from, to, consumers: [], cost: this._policy.estimateConversionCost(from, to, value.type), benefit: 0 };
        groups.set(key, g);
      }
      g.consumers.push({ consumer, operandIdx });
      const capable = this.target.layoutAwareOps && this.target.layoutAwareOps.has(consumer.opName);
      g.benefit += capable ? this._policy.estimateBenefit(consumer, value.type, 1) : 0;
    }

    let totalCost = 0, totalBenefit = 0;
    const keep = [];
    for (const g of groups.values()) {
      if (g.benefit < g.cost) continue;
      keep.push(g);
      totalCost += g.cost;
      totalBenefit += g.benefit;
    }
    if (keep.length === 0 || totalCost > totalBenefit) return PassResult.UNCHANGED;

    for (const g of keep) {
      const srcOrder = g.from instanceof Layout ? g.from.order : Array.from({ length: g.value.type.rank }, (_, k) => k);
      const dstOrder = g.to instanceof Layout ? g.to.order : Array.from({ length: g.value.type.rank }, (_, k) => k);
      const resultType = new TensorType(g.value.type.shape, g.value.type.dtype, g.to);
      const transformOp = new Operation('layout_transform', [g.value], [resultType], {
        src_layout: [...srcOrder],
        dst_layout: [...dstOrder]
      });
      const defOp = g.value.definingOp;
      if (defOp && defOp.parentBlock) {
        defOp.parentBlock.insertAfter(transformOp, defOp);
      } else if (g.consumers[0].consumer.parentBlock) {
        g.consumers[0].consumer.parentBlock.insertBefore(transformOp, g.consumers[0].consumer);
      }
      const tr = transformOp.getResult(0);
      for (const c of g.consumers) c.consumer.replaceOperand(c.operandIdx, tr);
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        conversions: result.conversions.length,
        uniqueTransforms: keep.length,
        level: TraceLevel.DEBUG,
      });
    }

    return PassResult.CHANGED;
  }
}

function valueLayoutKey(value, from, to) {
  const vid = value.id;
  const fh = from.hash ? from.hash() : 0;
  const th = to.hash ? to.hash() : 0;
  return `${vid}:${fh}:${th}`;
}
