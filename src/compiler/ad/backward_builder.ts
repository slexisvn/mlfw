import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { nameLocation } from '../ir/location.js';
import type { Location } from '../ir/location.js';

import { UseDefAnalysis } from '../analysis/use_def.js';
import { readValues } from '../ir/graph/graph_algorithms.js';
import { GradAccumulator, gradOrZero } from './grad_accumulator.js';
import { getVJPRule, isGradientBarrier, requireVJPRuleOrBarrier, getRegionVJP } from './vjp_registry.js';
import type { Explain } from '../passes/explain.js';
import { regionFreeVars } from '../ir/graph/graph_algorithms.js';
import { REGION_CONTROL_FLOW } from './control_flow_ops.js';

import { TensorType } from '../ir/graph/types.js';
import type { IRType, Shape } from '../ir/graph/types.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { TensorValue } from './vjp_registry.js';
import type { CheckpointPolicy } from './checkpoint_policy.js';
import type { RematPolicy } from './remat_policy.js';

export { REGION_CONTROL_FLOW };

export type ResolveValueFn = (v: Value) => Value;

function cloneOpWithRegions(builder: IRBuilder, op: Operation, resolve: ResolveValueFn): Operation {
  const remap = new Map<Value, Value>();
  for (const v of readValues(op)) {
    const mapped = resolve(v);
    if (mapped !== v) remap.set(v, mapped);
  }
  return builder._insert(op.clone(remap));
}

export type BackpropOptions = {
  accumulator: GradAccumulator;
  builder: IRBuilder;
  needsGrad: ReadonlySet<number>;
  resolveValue: ResolveValueFn;
  handleRegionOp?: ((op: Operation) => boolean) | null;
  explain?: Explain | null;
};

export type BackwardBuilderOpts = Readonly<{
  rematPolicy?: RematPolicy | null;
  checkpointPolicy?: CheckpointPolicy | null;
  scanCheckpoint?: unknown;
  explain?: Explain | null;
}>;

export type BackwardBuildResult = {
  backwardFunc: GraphFunction;
  savedValues: Value[];
  gradInputIndices: number[];
};

type SavedValueInfo = { savedValues: Value[]; savedValueIndices: Map<number, number> };

const GRADIENT_TAG = 'grad';

const FALLBACK_REMAT_OPS = new Set(['neg', 'abs', 'sign', 'floor', 'ceil']);

export function gradientLocation(loc: Location | null): Location | null {
  return loc === null ? null : nameLocation(GRADIENT_TAG, loc);
}

export function backpropOps(orderedOps: readonly Operation[], { accumulator, builder, needsGrad, resolveValue, handleRegionOp = null, explain = null }: BackpropOptions): void {
  for (let i = orderedOps.length - 1; i >= 0; i--) {
    const op = orderedOps[i];
    if (op.opName === 'return' || op.opName === 'constant') continue;

    const hasGradResult = op.results.some(r => needsGrad.has(r.id));
    if (!hasGradResult) continue;

    const gradOuts: (Value | null)[] = [];
    for (let r = 0; r < op.numResults; r++) gradOuts.push(accumulator.get(op.getResult(r).id));
    if (gradOuts.every(g => g === null)) continue;

    if (handleRegionOp && handleRegionOp(op)) continue;

    const rule = requireVJPRuleOrBarrier(op.opName);
    if (!rule) continue;

    if (explain) {
      if (isGradientBarrier(op.opName)) {
        explain(op.opName, 'gradient stops here',
          'this op is declared a gradient barrier, so nothing downstream of it contributes to the input gradients');
      } else {
        explain(op.opName, 'chain rule applied backwards',
          'its VJP rule turns the gradient of its output into a gradient for each operand that needs one');
      }
    }

    const operandValues = new Array<Value>(op.numOperands);
    for (let o = 0; o < op.numOperands; o++) operandValues[o] = resolveValue(op.getOperand(o));
    const resultValues = new Array<Value>(op.numResults);
    for (let r = 0; r < op.numResults; r++) resultValues[r] = resolveValue(op.getResult(r));

    const full = (value: number, type: TensorType) => builder.broadcast(builder.scalarConstant(value, type.dtype).getResult(0), type.shape, []).getResult(0) as TensorValue;
    builder.withLocation(gradientLocation(op.loc), () => {
      const gradInputs = rule({ builder, op, operands: operandValues as TensorValue[], results: resultValues as TensorValue[], gradOutputs: gradOuts as (TensorValue | null)[], attrs: op.attributes, full });
      if (!gradInputs) return;

      for (let o = 0; o < op.numOperands; o++) {
        if (o >= gradInputs.length || !gradInputs[o]) continue;
        const operandVal = op.getOperand(o);
        if (!needsGrad.has(operandVal.id)) continue;
        accumulator.accumulate(operandVal.id, reduceGradToOperandShape(builder, gradInputs[o] as Value, (operandVal.type as TensorType).shape));
      }
    });
  }
}

export function computeGradReachability(func: GraphFunction, topoOrder: readonly Operation[]): Set<number> {
  const needsGrad = new Set<number>();
  const returnOp = func.getReturnOp() as Operation;

  for (const val of returnOp.operands) {
    needsGrad.add(val.id);
  }

  for (let i = topoOrder.length - 1; i >= 0; i--) {
    const op = topoOrder[i];
    if (op.opName === 'return') continue;

    const hasGradResult = op.results.some(r => needsGrad.has(r.id));
    if (!hasGradResult) continue;

    if (REGION_CONTROL_FLOW.has(op.opName)) {
      for (let o = 0; o < op.numOperands; o++) needsGrad.add(op.getOperand(o).id);
      for (const fv of regionControlFlowFreeVars(op)) needsGrad.add(fv.id);
      continue;
    }

    if (!getVJPRule(op.opName)) continue;
    if (isGradientBarrier(op.opName)) continue;

    for (let o = 0; o < op.numOperands; o++) {
      needsGrad.add(op.getOperand(o).id);
    }
  }

  return needsGrad;
}

function regionControlFlowFreeVars(op: Operation): Value[] {
  const out: Value[] = [];
  for (const region of op.regions) {
    if (region.blocks[0]) out.push(...regionFreeVars(region.blocks[0]));
  }
  return out;
}

export function reduceGradToOperandShape(builder: IRBuilder, grad: Value, targetShape: Shape): Value {
  const gradShape = (grad.type as TensorType).shape;
  if (gradShape.length === targetShape.length && gradShape.every((d, i) => d === targetShape[i])) {
    return grad;
  }
  const nExtra = gradShape.length - targetShape.length;
  const dims: number[] = [];
  for (let i = 0; i < nExtra; i++) dims.push(i);
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && gradShape[nExtra + i] !== 1) dims.push(nExtra + i);
  }
  let g = grad;
  if (dims.length > 0) {
    const init = builder.scalarConstant(0, (grad.type as TensorType).dtype).getResult(0);
    g = builder.reduce(grad, init, dims, 'sum').getResult(0);
  }
  const gShape = (g.type as TensorType).shape;
  if (!(gShape.length === targetShape.length && gShape.every((d, i) => d === targetShape[i]))) {
    g = builder.reshape(g, targetShape).getResult(0);
  }
  return g;
}

export class BackwardGraphBuilder {
  private _rematPolicy: RematPolicy | null;
  private _checkpointPolicy: CheckpointPolicy | null;
  private _scanCheckpoint: unknown;
  private _explain: Explain | null;

  constructor(opts: BackwardBuilderOpts = {}) {
    this._rematPolicy = opts.rematPolicy || null;
    this._checkpointPolicy = opts.checkpointPolicy || null;
    this._scanCheckpoint = opts.scanCheckpoint || null;
    this._explain = opts.explain || null;
  }

  build(forwardFunc: GraphFunction): BackwardBuildResult {
    if (this._checkpointPolicy) {
      return this._buildCheckpointed(forwardFunc);
    }
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;

    const returnOp = forwardFunc.getReturnOp();
    if (!returnOp) throw new Error('Forward function has no return op');

    const forwardOutputs = returnOp.operands;
    const forwardInputs = forwardFunc.args;

    const needsGrad = computeGradReachability(forwardFunc, topoOrder);
    const { savedValues, savedValueIndices } = this._identifySavedValues(topoOrder, needsGrad, forwardInputs);

    const gradOutputTypes = forwardOutputs.map(v => v.type);
    const savedTypes = savedValues.map(v => v.type);
    const inputTypes = [...gradOutputTypes, ...savedTypes];

    const gradInputTypes: IRType[] = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        gradInputTypes.push(forwardInputs[i].type);
      }
    }

    const bwdFunc = new GraphFunction(
      `backward_${forwardFunc.name}`,
      inputTypes,
      gradInputTypes
    );

    const builder = new IRBuilder(bwdFunc);
    const bwdArgs = bwdFunc.args;

    const gradOutputArgs = bwdArgs.slice(0, gradOutputTypes.length);
    const savedArgs = bwdArgs.slice(gradOutputTypes.length);

    const valueMap = new Map<number, Value>();
    for (let i = 0; i < savedValues.length; i++) {
      valueMap.set(savedValues[i].id, savedArgs[i]);
    }
    for (let i = 0; i < forwardInputs.length; i++) {
      if (savedValueIndices.has(forwardInputs[i].id)) {
        valueMap.set(forwardInputs[i].id, savedArgs[savedValueIndices.get(forwardInputs[i].id) as number]);
      }
    }

    const accumulator = new GradAccumulator(builder);

    for (let i = 0; i < forwardOutputs.length; i++) {
      const outputVal = forwardOutputs[i];
      accumulator.accumulate(outputVal.id, gradOutputArgs[i]);
    }

    backpropOps(topoOrder, {
      accumulator, builder, needsGrad, explain: this._explain,
      resolveValue: (v: Value) => this._materialize(v, valueMap, builder),
      handleRegionOp: (op: Operation) => {
        const regionFn = getRegionVJP(op.opName);
        if (!regionFn) return false;
        (regionFn as unknown as (op: Operation, ctx: unknown) => void)(op, { accumulator, builder, materialize: (v: Value) => this._materialize(v, valueMap, builder), needsGrad, scanCheckpoint: this._scanCheckpoint });
        return true;
      },
    });

    const returnValues: Value[] = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        returnValues.push(gradOrZero(builder, forwardInputs[i], accumulator));
      }
    }

    builder.returnOp(returnValues);

    if (this._explain) {
      this._explain(bwdFunc.name, 'built as a separate function',
        'the backward reads the gradient of every forward output plus whatever forward values it still needs, and returns one gradient per input that asked for one',
        { gradOutputs: gradOutputTypes.length, savedValues: savedValues.length, inputGradients: returnValues.length });
    }

    return {
      backwardFunc: bwdFunc,
      savedValues,
      gradInputIndices: this._getGradInputIndices(forwardInputs, needsGrad),
    };
  }

  _materialize(rootVal: Value, valueMap: Map<number, Value>, builder: IRBuilder): Value {
    if (valueMap.has(rootVal.id)) return valueMap.get(rootVal.id) as Value;
    if (!rootVal.definingOp) return rootVal;

    const onStack = new Set<number>([rootVal.id]);
    const stack: { val: Value; i: number }[] = [{ val: rootVal, i: 0 }];
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const val = frame.val;
      const defOp = val.definingOp;
      if (valueMap.has(val.id) || !defOp) {
        onStack.delete(val.id);
        stack.pop();
        continue;
      }
      const defReads = readValues(defOp);
      if (frame.i < defReads.length) {
        const operand = defReads[frame.i];
        frame.i++;
        if (operand.definingOp && !valueMap.has(operand.id) && !onStack.has(operand.id)) {
          onStack.add(operand.id);
          stack.push({ val: operand, i: 0 });
        }
        continue;
      }
      const cloned = cloneOpWithRegions(builder, defOp, (v) => valueMap.get(v.id) ?? v);
      for (let r = 0; r < defOp.numResults; r++) {
        valueMap.set(defOp.getResult(r).id, cloned.getResult(r));
      }
      onStack.delete(val.id);
      stack.pop();
    }

    return valueMap.has(rootVal.id) ? valueMap.get(rootVal.id) as Value : rootVal;
  }

  _identifySavedValues(topoOrder: readonly Operation[], needsGrad: ReadonlySet<number>, forwardInputs: readonly Value[]): SavedValueInfo {
    const savedValues: Value[] = [];
    const savedValueIndices = new Map<number, number>();
    const inputIds = new Set(forwardInputs.map(v => v.id));

    for (const op of topoOrder) {
      if (op.opName === 'return' || op.opName === 'constant') continue;

      const rule = getVJPRule(op.opName);
      if (!rule) continue;

      const hasGradResult = op.results.some(r => needsGrad.has(r.id));
      if (!hasGradResult) continue;

      for (let o = 0; o < op.numOperands; o++) {
        const val = op.getOperand(o);
        if (inputIds.has(val.id) && !savedValueIndices.has(val.id)) {
          savedValueIndices.set(val.id, savedValues.length);
          savedValues.push(val);
        }
      }

      for (let r = 0; r < op.numResults; r++) {
        const val = op.getResult(r);
        if (needsGrad.has(val.id) && !savedValueIndices.has(val.id)) {
          if (this._shouldSaveResult(op)) {
            savedValueIndices.set(val.id, savedValues.length);
            savedValues.push(val);
          }
        }
      }
    }

    // A non-saved op consumed by the backward is rematerialized, which recurses
    // through its operands until it hits a saved value or a block-arg. Any forward
    // INPUT block-arg reached this way (e.g. index_select's indices come from
    // select(din)) would otherwise be referenced as a free variable that the graph
    // partitioner cannot map. Save those inputs so the backward func declares them.
    const savedIds = new Set(savedValueIndices.keys());
    const inputById = new Map<number, Value>(forwardInputs.map(v => [v.id, v]));
    const seen = new Set<number>();
    const collect = (val: Value): void => {
      if (savedIds.has(val.id) || seen.has(val.id)) return;
      seen.add(val.id);
      const defOp = val.definingOp;
      if (!defOp) {
        if (inputIds.has(val.id) && !savedValueIndices.has(val.id)) {
          savedValueIndices.set(val.id, savedValues.length);
          savedValues.push(inputById.get(val.id) as Value);
        }
        return;
      }
      for (let o = 0; o < defOp.numOperands; o++) collect(defOp.getOperand(o));
    };
    for (const op of topoOrder) {
      if (op.opName === 'return' || op.opName === 'constant') continue;
      if (!op.results.some(r => needsGrad.has(r.id))) continue;
      if (REGION_CONTROL_FLOW.has(op.opName)) {
        for (const v of op.operands) collect(v);
        for (const fv of regionControlFlowFreeVars(op)) collect(fv);
        continue;
      }
      if (!getVJPRule(op.opName)) continue;
      for (let o = 0; o < op.numOperands; o++) collect(op.getOperand(o));
      for (let r = 0; r < op.numResults; r++) collect(op.getResult(r));
    }

    return { savedValues, savedValueIndices };
  }

  _shouldSaveResult(op: Operation): boolean {
    const save = this._rematPolicy
      ? !this._rematPolicy.shouldRematerialize(op)
      : !FALLBACK_REMAT_OPS.has(op.opName);

    if (this._explain) {
      this._explain(op.opName, save ? 'kept alive for the backward' : 'recomputed in the backward',
        save
          ? 'the backward needs this value and recomputing it would cost more than the memory it occupies until then'
          : 'this op is cheap enough that running it again in the backward beats holding its result live across the whole forward');
    }

    return save;
  }

  _getGradInputIndices(forwardInputs: readonly Value[], needsGrad: ReadonlySet<number>): number[] {
    const indices: number[] = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        indices.push(i);
      }
    }
    return indices;
  }

  _buildCheckpointed(forwardFunc: GraphFunction): BackwardBuildResult {
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;

    const returnOp = forwardFunc.getReturnOp();
    if (!returnOp) throw new Error('Forward function has no return op');

    const forwardOutputs = returnOp.operands;
    const forwardInputs = forwardFunc.args;

    const needsGrad = computeGradReachability(forwardFunc, topoOrder);
    const segments = (this._checkpointPolicy as CheckpointPolicy).segment(topoOrder, forwardFunc);

    const savedValueSet = new Set<number>();
    const savedValues: Value[] = [];
    const savedValueIndices = new Map<number, number>();

    const valueById = new Map<number, Value>();
    for (const op of topoOrder) {
      for (let r = 0; r < op.numResults; r++) {
        const res = op.getResult(r);
        valueById.set(res.id, res);
      }
    }

    const inputIds = new Set(forwardInputs.map(v => v.id));
    for (const input of forwardInputs) {
      if (needsGrad.has(input.id) && !savedValueSet.has(input.id)) {
        savedValueSet.add(input.id);
        savedValueIndices.set(input.id, savedValues.length);
        savedValues.push(input);
      }
    }

    for (const seg of segments) {
      for (const valId of seg.boundaryInputs) {
        if (!inputIds.has(valId) && !savedValueSet.has(valId)) {
          savedValueSet.add(valId);
          const val = valueById.get(valId) || null;
          if (val) {
            savedValueIndices.set(valId, savedValues.length);
            savedValues.push(val);
          }
        }
      }
      for (const valId of seg.boundaryOutputs) {
        if (!savedValueSet.has(valId)) {
          savedValueSet.add(valId);
          const val = valueById.get(valId) || null;
          if (val) {
            savedValueIndices.set(valId, savedValues.length);
            savedValues.push(val);
          }
        }
      }
    }

    for (const output of forwardOutputs) {
      if (!savedValueSet.has(output.id)) {
        savedValueSet.add(output.id);
        savedValueIndices.set(output.id, savedValues.length);
        savedValues.push(output);
      }
    }

    const gradOutputTypes = forwardOutputs.map(v => v.type);
    const savedTypes = savedValues.map(v => v.type);
    const inputTypes = [...gradOutputTypes, ...savedTypes];

    const gradInputTypes: IRType[] = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        gradInputTypes.push(forwardInputs[i].type);
      }
    }

    const bwdFunc = new GraphFunction(
      `backward_${forwardFunc.name}`,
      inputTypes,
      gradInputTypes
    );

    const builder = new IRBuilder(bwdFunc);
    const bwdArgs = bwdFunc.args;

    const gradOutputArgs = bwdArgs.slice(0, gradOutputTypes.length);
    const savedArgs = bwdArgs.slice(gradOutputTypes.length);

    const savedValueMap = new Map<number, Value>();
    for (let i = 0; i < savedValues.length; i++) {
      savedValueMap.set(savedValues[i].id, savedArgs[i]);
    }

    const accumulator = new GradAccumulator(builder);

    for (let i = 0; i < forwardOutputs.length; i++) {
      accumulator.accumulate(forwardOutputs[i].id, gradOutputArgs[i]);
    }

    const constantMap = new Map<number, Value>();
    for (const op of topoOrder) {
      if (op.opName === 'constant') {
        const resultType = op.getResult(0).type;
        const cloned = builder._buildOp('constant', [], [resultType], new Map(op.attributes), null);
        constantMap.set(op.getResult(0).id, cloned.getResult(0));
      }
    }

    for (let s = segments.length - 1; s >= 0; s--) {
      const seg = segments[s];

      const recomputeMap = new Map<number, Value>();

      const resolve = (v: Value) => recomputeMap.get(v.id) || savedValueMap.get(v.id) || constantMap.get(v.id) || v;
      for (const op of seg.ops) {
        const cloned = cloneOpWithRegions(builder, op, resolve);

        for (let r = 0; r < op.numResults; r++) {
          recomputeMap.set(op.getResult(r).id, cloned.getResult(r));
        }
      }

      backpropOps(seg.ops, {
        accumulator, builder, needsGrad,
        resolveValue: resolve,
        handleRegionOp: (op: Operation) => {
          const regionFn = getRegionVJP(op.opName);
          if (!regionFn) return false;
          (regionFn as unknown as (op: Operation, ctx: unknown) => void)(op, {
            accumulator, builder, needsGrad, materialize: resolve, scanCheckpoint: this._scanCheckpoint,
          });
          return true;
        },
      });
    }

    const returnValues: Value[] = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        returnValues.push(gradOrZero(builder, forwardInputs[i], accumulator));
      }
    }

    builder.returnOp(returnValues);

    return {
      backwardFunc: bwdFunc,
      savedValues,
      gradInputIndices: this._getGradInputIndices(forwardInputs, needsGrad),
    };
  }

}
