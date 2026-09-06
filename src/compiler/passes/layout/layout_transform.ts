import { FunctionPass, PassResult } from '../pass.js';
import { Operation } from '../../ir/graph/operation.js';
import { producerLocation } from '../../ir/graph/op_location.js';
import { TensorType, Layout } from '../../ir/graph/types.js';
import type { AttrValue } from '../../ir/graph/types.js';
import { LayoutPolicy } from './layout_policy.js';
import { LayoutAnalysis } from '../../analysis/layout_analysis.js';
import { UseDefAnalysis } from '../../analysis/use_def.js';
import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import { registry } from '../../ir/graph/ops.js';
import { OpTrait } from '../../ir/graph/op_registry.js';
import { isTerminatorOp, isBroadcastOp, isConstantOp } from '../../ir/graph/op_traits.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Block } from '../../ir/graph/block.js';
import type { Value } from '../../ir/graph/value.js';
import type { AnalysisManager } from '../../analysis/analysis_manager.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { LayoutPolicyTarget } from './layout_policy.js';
import type { AttrRecord } from '../../ir/graph/builder.js';

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
      const srcOrder = g.from.order;
      const dstOrder = g.to.order;
      const resultType = new TensorType(vType.shape, vType.dtype, g.to);
      const attrs: AttrRecord = { src_layout: [...srcOrder], dst_layout: [...dstOrder] };
      if (g.from.block) attrs.src_block = [g.from.block.dim, g.from.block.factor];
      if (g.to.block) attrs.dst_block = [g.to.block.dim, g.to.block.factor];
      const transformOp = new Operation('layout_transform', [g.value], [resultType], attrs);
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

    propagateBlockedLayouts(graphFunc, this._policy as LayoutPolicy, explain);

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

function blockedLayoutOf(value: Value): Layout | null {
  const t = value.type;
  if (!(t instanceof TensorType) || !t.layout || !t.layout.isBlocked()) return null;
  return t.layout;
}

function tensorOperands(op: Operation): Value[] {
  const out: Value[] = [];
  for (let i = 0; i < op.numOperands; i++) {
    const v = op.getOperand(i);
    if (v.type instanceof TensorType) out.push(v);
  }
  return out;
}

function isUniformConstant(op: Operation): boolean {
  if (!isConstantOp(op.opName)) return false;
  const value = op.getAttr<AttrValue>('value');
  if (typeof value === 'number') return true;
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) return false;
  const data = value as ArrayLike<number>;
  for (let i = 1; i < data.length; i++) {
    if (data[i] !== data[0]) return false;
  }
  return true;
}

function isSplat(value: Value): boolean {
  const def = value.definingOp;
  if (!def) return false;
  if (isUniformConstant(def)) return true;
  if (!isBroadcastOp(def.opName)) return false;
  const src = def.getOperand(0).type;
  return src instanceof TensorType && (src.numel() === 1 || isSplat(def.getOperand(0)));
}

function isScalarOperand(value: Value): boolean {
  const t = value.type;
  return t instanceof TensorType && t.numel() === 1;
}

function isLayoutAgnostic(op: Operation): boolean {
  const def = registry.get(op.opName);
  if (!def || !def.hasTrait(OpTrait.ELEMENTWISE) || op.numResults !== 1) return false;
  const result = op.getResult(0).type;
  if (!(result instanceof TensorType)) return false;
  const operands = tensorOperands(op);
  if (operands.length === 0) return false;
  return operands.every(v => (v.type as TensorType).shapeEquals(result) || isScalarOperand(v));
}

function acceptsBlocked(op: Operation, operandIdx: number, layout: Layout, policy: LayoutPolicy): boolean {
  if (isTerminatorOp(op.opName)) return false;
  if (isLayoutAgnostic(op)) {
    return tensorOperands(op).every(v => isScalarOperand(v) || layout.equals(blockedLayoutOf(v)));
  }
  const pref = policy.getPreference(op);
  const wanted = pref ? pref.inputs[operandIdx] : null;
  return !!wanted && layout.equals(wanted);
}

function retypeResults(op: Operation, layout: Layout): void {
  for (let r = 0; r < op.numResults; r++) {
    const value = op.getResult(r);
    const t = value.type;
    if (t instanceof TensorType) value.type = new TensorType(t.shape, t.dtype, layout);
  }
}

function propagateBlockedLayouts(
  func: GraphFunction,
  policy: LayoutPolicy,
  explain: ReturnType<typeof explainer>
): void {
  for (const op of [...func.ops()]) {
    if (isTerminatorOp(op.opName)) continue;
    const pref = policy.getPreference(op);
    if (pref) {
      const out = pref.outputs[0];
      const wantsBlocked = out instanceof Layout && out.isBlocked();
      const operandsReady = pref.inputs.every((want, i) =>
        !want || (i < op.numOperands && want.equals(blockedLayoutOf(op.getOperand(i)) ?? (op.getOperand(i).type as TensorType).layout)));
      if (wantsBlocked && operandsReady) retypeResults(op, out);
      continue;
    }
    if (!isLayoutAgnostic(op)) continue;
    const operands = tensorOperands(op);
    const layout = operands.map(blockedLayoutOf).find(l => l !== null);
    if (!layout) continue;
    if (!operands.every(v => layout.equals(blockedLayoutOf(v)) || isScalarOperand(v) || isSplat(v))) continue;
    for (const v of operands) {
      if (!blockedLayoutOf(v) && !isScalarOperand(v)) retypeResults(v.definingOp as Operation, layout);
    }
    retypeResults(op, layout);
  }

  for (const op of [...func.ops()]) {
    for (let i = 0; i < op.numOperands; i++) {
      const value = op.getOperand(i);
      const layout = blockedLayoutOf(value);
      if (!layout || acceptsBlocked(op, i, layout, policy)) continue;
      const t = value.type as TensorType;
      const plain = new Layout(layout.order);
      const back = new Operation('layout_transform', [value], [new TensorType(t.shape, t.dtype, plain)], {
        src_layout: [...layout.order],
        dst_layout: [...plain.order],
        src_block: [layout.block!.dim, layout.block!.factor]
      } as AttrRecord);
      back.loc = producerLocation([value]);
      (op.parentBlock as Block).insertBefore(back, op);
      op.replaceOperand(i, back.getResult(0));
      if (explain) {
        explain(op.opName, `read back as [${plain.order.join(', ')}]`,
          'this op indexes its operand logically, so the blocked form has to be undone before it',
          { blockFactor: layout.block!.factor });
      }
    }
  }
}

function valueLayoutKey(value: Value, from: Layout, to: Layout): string {
  const vid = value.id;
  const fh = from.hash ? from.hash() : 0;
  const th = to.hash ? to.hash() : 0;
  return `${vid}:${fh}:${th}`;
}
