import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { producerLocation } from '../../ir/graph/op_location.js';
import { TensorType, Layout } from '../../ir/graph/types.js';
import { LayoutPolicy } from './layout_policy.js';
import { LayoutAnalysis } from '../../analysis/layout_analysis.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Block } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { LayoutPolicyTarget } from './layout_policy.js';

export type LayoutTransformConfig = { target?: LayoutPolicyTarget | null };
type ConversionConsumer = { consumer: Operation; operandIdx: number };
type ConversionGroup = {
  value: Value;
  from: Layout;
  to: Layout;
  consumers: ConversionConsumer[];
  cost: number;
  benefit: number;
};

function layoutSubject(group: ConversionGroup): string {
  const def = group.value.definingOp;
  return def ? def.opName : `arg%${group.value.id}`;
}

export class LayoutTransformPass extends FunctionPass {
  target: LayoutPolicyTarget | null;
  private _policy: LayoutPolicy | null;

  constructor(config: LayoutTransformConfig = {}) {
    super('LayoutTransformPass');
    this.requiredAnalyses = [UseDefAnalysis];
    this.target = config.target || null;
    this._policy = null;
  }

  override run(func: PassTarget, analysisManager?: AnalysisManager): PassResultValue {
    const graphFunc = func as GraphFunction;
    if (!this.target) return PassResult.UNCHANGED;

    if (!this._policy) {
      this._policy = new LayoutPolicy(this.target);
    }

    const useDef = this.getAnalysis(UseDefAnalysis, graphFunc, analysisManager);

    const result = LayoutAnalysis.compute(graphFunc, { useDef }, this._policy as never);

    if (result.conversions.length === 0) return PassResult.UNCHANGED;

    const groups = new Map<string, ConversionGroup>();
    for (const conv of result.conversions) {
      const { value, consumer, operandIdx, from, to } = conv;
      const key = valueLayoutKey(value, from, to);
      let g = groups.get(key);
      if (!g) {
        g = { value, from, to, consumers: [], cost: (this._policy as LayoutPolicy).estimateConversionCost(from, to, value.type), benefit: 0 };
        groups.set(key, g);
      }
      g.consumers.push({ consumer, operandIdx });
      const capable = this.target.layoutAwareOps && this.target.layoutAwareOps.has(consumer.opName);
      g.benefit += capable ? (this._policy as LayoutPolicy).estimateBenefit(consumer, value.type, 1) : 0;
    }

    const explain = explainer(this.trace, this.name);
    let totalCost = 0, totalBenefit = 0;
    const keep: ConversionGroup[] = [];
    for (const g of groups.values()) {
      if (g.benefit < g.cost) {
        if (explain) {
          explain(layoutSubject(g), 'left in its current layout',
            'the transpose this layout would need costs more than the ops reading it would gain',
            { conversionCost: g.cost, estimatedBenefit: g.benefit });
        }
        continue;
      }
      keep.push(g);
      totalCost += g.cost;
      totalBenefit += g.benefit;
    }
    if (keep.length === 0 || totalCost > totalBenefit) return PassResult.UNCHANGED;

    for (const g of keep) {
      const vType = g.value.type as TensorType;
      const srcOrder = g.from instanceof Layout ? g.from.order : Array.from({ length: vType.rank }, (_, k) => k);
      const dstOrder = g.to instanceof Layout ? g.to.order : Array.from({ length: vType.rank }, (_, k) => k);
      const resultType = new TensorType(vType.shape, vType.dtype, g.to);
      const transformOp = new Operation('layout_transform', [g.value], [resultType], {
        src_layout: [...srcOrder],
        dst_layout: [...dstOrder]
      });
      transformOp.loc = producerLocation([g.value]);
      const defOp = g.value.definingOp;
      if (defOp && defOp.parentBlock) {
        defOp.parentBlock.insertAfter(transformOp, defOp);
      } else if (g.consumers[0].consumer.parentBlock) {
        (g.consumers[0].consumer.parentBlock as Block).insertBefore(transformOp, g.consumers[0].consumer);
      }
      const tr = transformOp.getResult(0);
      for (const c of g.consumers) c.consumer.replaceOperand(c.operandIdx, tr);
      if (explain) {
        explain(layoutSubject(g), `laid out as [${dstOrder.join(', ')}]`,
          'the ops reading this value run faster in that order by more than the inserted transpose costs',
          { conversionCost: g.cost, estimatedBenefit: g.benefit, readers: g.consumers.length });
      }
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

function valueLayoutKey(value: Value, from: Layout, to: Layout): string {
  const vid = value.id;
  const fh = from.hash ? from.hash() : 0;
  const th = to.hash ? to.hash() : 0;
  return `${vid}:${fh}:${th}`;
}
