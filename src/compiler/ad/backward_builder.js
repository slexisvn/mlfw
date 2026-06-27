import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';

import { UseDefAnalysis } from '../analysis/use_def.js';
import { GradAccumulator, gradOrZero } from './grad_accumulator.js';
import { getVJPRule, isGradientBarrier, requireVJPRuleOrBarrier, getRegionVJP } from './vjp_registry.js';
import { buildScanBackward, buildCondBackward, regionFreeVars } from './scan_backward.js';
import { REGION_CONTROL_FLOW } from './control_flow_ops.js';

export { REGION_CONTROL_FLOW };

const FALLBACK_REMAT_OPS = new Set(['neg', 'abs', 'sign', 'floor', 'ceil']);

export function backpropOps(orderedOps, { accumulator, builder, needsGrad, resolveValue, handleRegionOp = null }) {
  for (let i = orderedOps.length - 1; i >= 0; i--) {
    const op = orderedOps[i];
    if (op.opName === 'return' || op.opName === 'constant') continue;

    const hasGradResult = op.results.some(r => needsGrad.has(r.id));
    if (!hasGradResult) continue;

    const gradOuts = [];
    for (let r = 0; r < op.numResults; r++) gradOuts.push(accumulator.get(op.getResult(r).id));
    if (gradOuts.every(g => g === null)) continue;

    if (handleRegionOp && handleRegionOp(op)) continue;

    const rule = requireVJPRuleOrBarrier(op.opName);
    if (!rule) continue;

    const operandValues = new Array(op.numOperands);
    for (let o = 0; o < op.numOperands; o++) operandValues[o] = resolveValue(op.getOperand(o));
    const resultValues = new Array(op.numResults);
    for (let r = 0; r < op.numResults; r++) resultValues[r] = resolveValue(op.getResult(r));

    const gradInputs = rule({ builder, op, operands: operandValues, results: resultValues, gradOutputs: gradOuts, attrs: op.attributes });
    if (!gradInputs) continue;

    for (let o = 0; o < op.numOperands; o++) {
      if (o >= gradInputs.length || !gradInputs[o]) continue;
      const operandVal = op.getOperand(o);
      if (!needsGrad.has(operandVal.id)) continue;
      accumulator.accumulate(operandVal.id, reduceGradToOperandShape(builder, gradInputs[o], operandVal.type.shape));
    }
  }
}

function regionControlFlowFreeVars(op) {
  const out = [];
  for (const region of op.regions) {
    if (region.blocks[0]) out.push(...regionFreeVars(region.blocks[0]));
  }
  return out;
}

export function reduceGradToOperandShape(builder, grad, targetShape) {
  const gradShape = grad.type.shape;
  if (gradShape.length === targetShape.length && gradShape.every((d, i) => d === targetShape[i])) {
    return grad;
  }
  const nExtra = gradShape.length - targetShape.length;
  const dims = [];
  for (let i = 0; i < nExtra; i++) dims.push(i);
  for (let i = 0; i < targetShape.length; i++) {
    if (targetShape[i] === 1 && gradShape[nExtra + i] !== 1) dims.push(nExtra + i);
  }
  let g = grad;
  if (dims.length > 0) {
    const init = builder.scalarConstant(0, grad.type.dtype).getResult(0);
    g = builder.reduce(grad, init, dims, 'sum').getResult(0);
  }
  const gShape = g.type.shape;
  if (!(gShape.length === targetShape.length && gShape.every((d, i) => d === targetShape[i]))) {
    g = builder.reshape(g, targetShape).getResult(0);
  }
  return g;
}

export class BackwardGraphBuilder {
  constructor(opts = {}) {
    this._rematPolicy = opts.rematPolicy || null;
    this._checkpointPolicy = opts.checkpointPolicy || null;
    this._scanCheckpoint = opts.scanCheckpoint || null;
  }

  build(forwardFunc) {
    if (this._checkpointPolicy) {
      return this._buildCheckpointed(forwardFunc);
    }
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;

    const returnOp = forwardFunc.getReturnOp();
    if (!returnOp) throw new Error('Forward function has no return op');

    const forwardOutputs = returnOp.operands;
    const forwardInputs = forwardFunc.args;

    const needsGrad = this._computeGradReachability(forwardFunc, topoOrder);
    const { savedValues, savedValueIndices } = this._identifySavedValues(topoOrder, needsGrad, forwardInputs);

    const gradOutputTypes = forwardOutputs.map(v => v.type);
    const savedTypes = savedValues.map(v => v.type);
    const inputTypes = [...gradOutputTypes, ...savedTypes];

    const gradInputTypes = [];
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

    const valueMap = new Map();
    for (let i = 0; i < savedValues.length; i++) {
      valueMap.set(savedValues[i].id, savedArgs[i]);
    }
    for (let i = 0; i < forwardInputs.length; i++) {
      if (savedValueIndices.has(forwardInputs[i].id)) {
        valueMap.set(forwardInputs[i].id, savedArgs[savedValueIndices.get(forwardInputs[i].id)]);
      }
    }

    const accumulator = new GradAccumulator(builder);

    for (let i = 0; i < forwardOutputs.length; i++) {
      const outputVal = forwardOutputs[i];
      accumulator.accumulate(outputVal.id, gradOutputArgs[i]);
    }

    backpropOps(topoOrder, {
      accumulator, builder, needsGrad,
      resolveValue: (v) => this._materialize(v, valueMap, builder),
      handleRegionOp: (op) => {
        const regionFn = getRegionVJP(op.opName);
        if (!regionFn) return false;
        regionFn(op, { accumulator, builder, materialize: (v) => this._materialize(v, valueMap, builder), needsGrad, scanCheckpoint: this._scanCheckpoint });
        return true;
      },
    });

    const returnValues = [];
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

  _materialize(rootVal, valueMap, builder) {
    if (valueMap.has(rootVal.id)) return valueMap.get(rootVal.id);
    if (!rootVal.definingOp) return rootVal;

    const onStack = new Set([rootVal.id]);
    const stack = [{ val: rootVal, i: 0 }];
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const val = frame.val;
      const defOp = val.definingOp;
      if (valueMap.has(val.id) || !defOp) {
        onStack.delete(val.id);
        stack.pop();
        continue;
      }
      if (frame.i < defOp.numOperands) {
        const operand = defOp.getOperand(frame.i);
        frame.i++;
        if (operand.definingOp && !valueMap.has(operand.id) && !onStack.has(operand.id)) {
          onStack.add(operand.id);
          stack.push({ val: operand, i: 0 });
        }
        continue;
      }
      const operands = new Array(defOp.numOperands);
      for (let o = 0; o < defOp.numOperands; o++) {
        const ov = defOp.getOperand(o);
        operands[o] = valueMap.has(ov.id) ? valueMap.get(ov.id) : ov;
      }
      const resultTypes = defOp.results.map(r => r.type);
      const cloned = builder._buildOp(defOp.opName, operands, resultTypes, new Map(defOp.attributes), null);
      for (let r = 0; r < defOp.numResults; r++) {
        valueMap.set(defOp.getResult(r).id, cloned.getResult(r));
      }
      onStack.delete(val.id);
      stack.pop();
    }

    return valueMap.has(rootVal.id) ? valueMap.get(rootVal.id) : rootVal;
  }

  _computeGradReachability(func, topoOrder) {
    const needsGrad = new Set();
    const returnOp = func.getReturnOp();

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

  _identifySavedValues(topoOrder, needsGrad, forwardInputs) {
    const savedValues = [];
    const savedValueIndices = new Map();
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
    const inputById = new Map(forwardInputs.map(v => [v.id, v]));
    const seen = new Set();
    const collect = (val) => {
      if (savedIds.has(val.id) || seen.has(val.id)) return;
      seen.add(val.id);
      const defOp = val.definingOp;
      if (!defOp) {
        if (inputIds.has(val.id) && !savedValueIndices.has(val.id)) {
          savedValueIndices.set(val.id, savedValues.length);
          savedValues.push(inputById.get(val.id));
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

  _shouldSaveResult(op) {
    if (this._rematPolicy) {
      return !this._rematPolicy.shouldRematerialize(op);
    }
    return !FALLBACK_REMAT_OPS.has(op.opName);
  }

  _getGradInputIndices(forwardInputs, needsGrad) {
    const indices = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        indices.push(i);
      }
    }
    return indices;
  }

  _buildCheckpointed(forwardFunc) {
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;

    const returnOp = forwardFunc.getReturnOp();
    if (!returnOp) throw new Error('Forward function has no return op');

    const forwardOutputs = returnOp.operands;
    const forwardInputs = forwardFunc.args;

    const needsGrad = this._computeGradReachability(forwardFunc, topoOrder);
    const segments = this._checkpointPolicy.segment(topoOrder, forwardFunc);

    for (const seg of segments) {
      for (const op of seg.ops) {
        if (REGION_CONTROL_FLOW.has(op.opName)) {
          throw new Error(`Checkpointed backward does not support region control-flow op '${op.opName}'; build the backward without a checkpointPolicy, which differentiates scan/if via buildScanBackward/buildCondBackward.`);
        }
      }
    }

    const savedValueSet = new Set();
    const savedValues = [];
    const savedValueIndices = new Map();

    const valueById = new Map();
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

    const gradInputTypes = [];
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

    const savedValueMap = new Map();
    for (let i = 0; i < savedValues.length; i++) {
      savedValueMap.set(savedValues[i].id, savedArgs[i]);
    }

    const accumulator = new GradAccumulator(builder);

    for (let i = 0; i < forwardOutputs.length; i++) {
      accumulator.accumulate(forwardOutputs[i].id, gradOutputArgs[i]);
    }

    const constantMap = new Map();
    for (const op of topoOrder) {
      if (op.opName === 'constant') {
        const resultType = op.getResult(0).type;
        const cloned = builder._buildOp('constant', [], [resultType], new Map(op.attributes), null);
        constantMap.set(op.getResult(0).id, cloned.getResult(0));
      }
    }

    for (let s = segments.length - 1; s >= 0; s--) {
      const seg = segments[s];

      const recomputeMap = new Map();

      for (const op of seg.ops) {
        const newOperands = new Array(op.numOperands);
        for (let o = 0; o < op.numOperands; o++) {
          const origVal = op.getOperand(o);
          const mapped = recomputeMap.get(origVal.id) ||
                         savedValueMap.get(origVal.id) ||
                         constantMap.get(origVal.id);
          newOperands[o] = mapped || origVal;
        }

        const resultTypes = op.results.map(r => r.type);
        const cloned = builder._buildOp(op.opName, newOperands, resultTypes, new Map(op.attributes), null);

        for (let r = 0; r < op.numResults; r++) {
          recomputeMap.set(op.getResult(r).id, cloned.getResult(r));
        }
      }

      backpropOps(seg.ops, {
        accumulator, builder, needsGrad,
        resolveValue: (v) => recomputeMap.get(v.id) || savedValueMap.get(v.id) || constantMap.get(v.id) || v,
      });
    }

    const returnValues = [];
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
