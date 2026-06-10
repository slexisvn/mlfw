import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { TensorType } from '../ir/graph/types.js';
import { UseDefAnalysis } from '../analysis/use_def.js';
import { GradAccumulator } from './grad_accumulator.js';
import { getVJPRule } from './vjp_registry.js';

export class BackwardGraphBuilder {
  constructor(opts = {}) {
    this._rematPolicy = opts.rematPolicy || null;
    this._checkpointPolicy = opts.checkpointPolicy || null;
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

    for (let i = topoOrder.length - 1; i >= 0; i--) {
      const op = topoOrder[i];
      if (op.opName === 'return') continue;
      if (op.opName === 'constant') continue;

      const hasGradResult = op.results.some(r => needsGrad.has(r.id));
      if (!hasGradResult) continue;

      const gradOuts = [];
      for (let r = 0; r < op.numResults; r++) {
        gradOuts.push(accumulator.get(op.getResult(r).id));
      }

      if (gradOuts.every(g => g === null)) continue;

      const rule = getVJPRule(op.opName);
      if (!rule) continue;

      const operandValues = new Array(op.numOperands);
      for (let o = 0; o < op.numOperands; o++) {
        operandValues[o] = this._materialize(op.getOperand(o), valueMap, builder);
      }

      const resultValues = new Array(op.numResults);
      for (let r = 0; r < op.numResults; r++) {
        resultValues[r] = this._materialize(op.getResult(r), valueMap, builder);
      }

      const ctx = {
        builder,
        op,
        operands: operandValues,
        results: resultValues,
        gradOutputs: gradOuts,
        attrs: op.attributes,
      };

      const gradInputs = rule(ctx);
      if (!gradInputs) continue;

      for (let o = 0; o < op.numOperands; o++) {
        if (o >= gradInputs.length || !gradInputs[o]) continue;
        const operandVal = op.getOperand(o);
        if (!needsGrad.has(operandVal.id)) continue;
        accumulator.accumulate(operandVal.id, gradInputs[o]);
      }
    }

    const returnValues = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        const grad = accumulator.get(forwardInputs[i].id);
        if (grad) {
          returnValues.push(grad);
        } else {
          const zeroConst = builder.scalarConstant(0, forwardInputs[i].type.dtype).getResult(0);
          const zeroBroadcast = builder.broadcast(zeroConst, forwardInputs[i].type.shape, []).getResult(0);
          returnValues.push(zeroBroadcast);
        }
      }
    }

    builder.returnOp(returnValues);

    return {
      backwardFunc: bwdFunc,
      savedValues,
      gradInputIndices: this._getGradInputIndices(forwardInputs, needsGrad),
    };
  }

  _materialize(fwdVal, valueMap, builder, active) {
    const mapped = valueMap.get(fwdVal.id);
    if (mapped !== undefined) return mapped;

    const defOp = fwdVal.definingOp;
    if (!defOp) return fwdVal;

    const seen = active || new Set();
    if (seen.has(fwdVal.id)) return fwdVal;
    seen.add(fwdVal.id);

    const operands = new Array(defOp.numOperands);
    for (let o = 0; o < defOp.numOperands; o++) {
      operands[o] = this._materialize(defOp.getOperand(o), valueMap, builder, seen);
    }

    const resultTypes = defOp.results.map(r => r.type);
    const cloned = builder._buildOp(defOp.opName, operands, resultTypes, new Map(defOp.attributes), null);

    for (let r = 0; r < defOp.numResults; r++) {
      valueMap.set(defOp.getResult(r).id, cloned.getResult(r));
    }

    return valueMap.get(fwdVal.id);
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

      if (!getVJPRule(op.opName)) continue;

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

    return { savedValues, savedValueIndices };
  }

  _shouldSaveResult(op) {
    if (this._rematPolicy) {
      return !this._rematPolicy.shouldRematerialize(op);
    }
    const name = op.opName;
    const alwaysRemat = new Set(['neg', 'abs', 'sign', 'floor', 'ceil']);
    return !alwaysRemat.has(name);
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

    const savedValueSet = new Set();
    const savedValues = [];
    const savedValueIndices = new Map();

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
          const val = this._findValue(topoOrder, valId);
          if (val) {
            savedValueIndices.set(valId, savedValues.length);
            savedValues.push(val);
          }
        }
      }
      for (const valId of seg.boundaryOutputs) {
        if (!savedValueSet.has(valId)) {
          savedValueSet.add(valId);
          const val = this._findValue(topoOrder, valId);
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

      for (let i = seg.ops.length - 1; i >= 0; i--) {
        const op = seg.ops[i];
        if (op.opName === 'return' || op.opName === 'constant') continue;

        const hasGradResult = op.results.some(r => needsGrad.has(r.id));
        if (!hasGradResult) continue;

        const gradOuts = [];
        for (let r = 0; r < op.numResults; r++) {
          gradOuts.push(accumulator.get(op.getResult(r).id));
        }

        if (gradOuts.every(g => g === null)) continue;

        const rule = getVJPRule(op.opName);
        if (!rule) continue;

        const operandValues = new Array(op.numOperands);
        for (let o = 0; o < op.numOperands; o++) {
          const fwdVal = op.getOperand(o);
          operandValues[o] = recomputeMap.get(fwdVal.id) ||
                             savedValueMap.get(fwdVal.id) ||
                             constantMap.get(fwdVal.id) ||
                             fwdVal;
        }

        const resultValues = new Array(op.numResults);
        for (let r = 0; r < op.numResults; r++) {
          const fwdRes = op.getResult(r);
          resultValues[r] = recomputeMap.get(fwdRes.id) ||
                            savedValueMap.get(fwdRes.id) ||
                            constantMap.get(fwdRes.id) ||
                            fwdRes;
        }

        const ctx = {
          builder,
          op,
          operands: operandValues,
          results: resultValues,
          gradOutputs: gradOuts,
          attrs: op.attributes,
        };

        const gradInputs = rule(ctx);
        if (!gradInputs) continue;

        for (let o = 0; o < op.numOperands; o++) {
          if (o >= gradInputs.length || !gradInputs[o]) continue;
          const operandVal = op.getOperand(o);
          if (!needsGrad.has(operandVal.id)) continue;
          accumulator.accumulate(operandVal.id, gradInputs[o]);
        }
      }
    }

    const returnValues = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      if (needsGrad.has(forwardInputs[i].id)) {
        const grad = accumulator.get(forwardInputs[i].id);
        if (grad) {
          returnValues.push(grad);
        } else {
          const zeroConst = builder.scalarConstant(0, forwardInputs[i].type.dtype).getResult(0);
          const zeroBroadcast = builder.broadcast(zeroConst, forwardInputs[i].type.shape, []).getResult(0);
          returnValues.push(zeroBroadcast);
        }
      }
    }

    builder.returnOp(returnValues);

    return {
      backwardFunc: bwdFunc,
      savedValues,
      gradInputIndices: this._getGradInputIndices(forwardInputs, needsGrad),
    };
  }

  _findValue(topoOrder, valueId) {
    for (const op of topoOrder) {
      for (let r = 0; r < op.numResults; r++) {
        if (op.getResult(r).id === valueId) return op.getResult(r);
      }
    }
    return null;
  }
}
