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

    const insertedTransforms = new Map();

    for (let i = 0; i < result.conversions.length; i++) {
      const conv = result.conversions[i];
      const { value, consumer, operandIdx, from, to } = conv;

      const key = valueLayoutKey(value, from, to);
      let transformResult = insertedTransforms.get(key);

      if (!transformResult) {
        const convCost = this._policy.estimateConversionCost(from, to, value.type);
        const useCount = value.uses ? [...value.uses()].length : 1;
        const benefit = this._policy.estimateBenefit(consumer, value.type, useCount);
        if (convCost > benefit) continue;

        const srcOrder = from instanceof Layout ? from.order : Array.from({ length: value.type.rank }, (_, k) => k);
        const dstOrder = to instanceof Layout ? to.order : Array.from({ length: value.type.rank }, (_, k) => k);

        const resultType = new TensorType(value.type.shape, value.type.dtype, to);
        const transformOp = new Operation('layout_transform', [value], [resultType], {
          src_layout: [...srcOrder],
          dst_layout: [...dstOrder]
        });

        const defOp = value.definingOp;
        if (defOp && defOp.parentBlock) {
          defOp.parentBlock.insertAfter(transformOp, defOp);
        } else if (consumer.parentBlock) {
          consumer.parentBlock.insertBefore(transformOp, consumer);
        }

        transformResult = transformOp.getResult(0);
        insertedTransforms.set(key, transformResult);
      }

      consumer.replaceOperand(operandIdx, transformResult);
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        conversions: result.conversions.length,
        uniqueTransforms: insertedTransforms.size,
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
