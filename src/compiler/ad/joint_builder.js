import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { UseDefAnalysis } from '../analysis/use_def.js';
import { GradAccumulator } from './grad_accumulator.js';
import { getVJPRule } from './vjp_registry.js';
import { RematPolicy } from './remat_policy.js';

export class JointGraphBuilder {
  constructor(opts = {}) {
    this._rematPolicy = opts.rematPolicy || new RematPolicy(opts.remat || {});
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

    const gradOutputTypes = forwardOutputs.map(v => v.type);
    const inputTypes = [...forwardFunc.inputTypes, ...gradOutputTypes];

    const outputTypes = [
      ...forwardFunc.outputTypes,
      ...forwardFunc.inputTypes,
    ];

    const jointFunc = new GraphFunction(
      `joint_${forwardFunc.name}`,
      inputTypes,
      outputTypes
    );

    const builder = new IRBuilder(jointFunc);
    const jointArgs = jointFunc.args;

    const fwdInputArgs = jointArgs.slice(0, forwardFunc.inputTypes.length);
    const gradOutputArgs = jointArgs.slice(forwardFunc.inputTypes.length);

    const valueMap = new Map();
    for (let i = 0; i < forwardInputs.length; i++) {
      valueMap.set(forwardInputs[i].id, fwdInputArgs[i]);
    }

    for (const op of topoOrder) {
      if (op.opName === 'return') continue;

      const newOperands = new Array(op.numOperands);
      for (let o = 0; o < op.numOperands; o++) {
        const origVal = op.getOperand(o);
        newOperands[o] = valueMap.get(origVal.id) || origVal;
      }

      const resultTypes = op.results.map(r => r.type);
      const cloned = builder._buildOp(op.opName, newOperands, resultTypes, new Map(op.attributes), null);

      for (let r = 0; r < op.numResults; r++) {
        valueMap.set(op.getResult(r).id, cloned.getResult(r));
      }
    }

    const fwdOutputValues = forwardOutputs.map(v => valueMap.get(v.id));

    const needsGrad = this._computeGradReachability(forwardFunc, topoOrder);
    const accumulator = new GradAccumulator(builder);

    for (let i = 0; i < forwardOutputs.length; i++) {
      accumulator.accumulate(forwardOutputs[i].id, gradOutputArgs[i]);
    }

    for (let i = topoOrder.length - 1; i >= 0; i--) {
      const op = topoOrder[i];
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
        const origVal = op.getOperand(o);
        operandValues[o] = valueMap.get(origVal.id) || origVal;
      }

      const resultValues = new Array(op.numResults);
      for (let r = 0; r < op.numResults; r++) {
        const origRes = op.getResult(r);
        resultValues[r] = valueMap.get(origRes.id) || origRes;
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

    const gradInputValues = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      const grad = accumulator.get(forwardInputs[i].id);
      if (grad) {
        gradInputValues.push(grad);
      } else {
        const zeroConst = builder.scalarConstant(0, forwardInputs[i].type.dtype).getResult(0);
        const zeroBroadcast = builder.broadcast(zeroConst, forwardInputs[i].type.shape, []).getResult(0);
        gradInputValues.push(zeroBroadcast);
      }
    }

    builder.returnOp([...fwdOutputValues, ...gradInputValues]);

    return {
      jointFunc,
      numForwardOutputs: forwardOutputs.length,
      numGradInputs: gradInputValues.length,
    };
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

  _buildCheckpointed(forwardFunc) {
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;

    const returnOp = forwardFunc.getReturnOp();
    if (!returnOp) throw new Error('Forward function has no return op');

    const forwardOutputs = returnOp.operands;
    const forwardInputs = forwardFunc.args;

    const gradOutputTypes = forwardOutputs.map(v => v.type);
    const inputTypes = [...forwardFunc.inputTypes, ...gradOutputTypes];

    const outputTypes = [
      ...forwardFunc.outputTypes,
      ...forwardFunc.inputTypes,
    ];

    const jointFunc = new GraphFunction(
      `joint_${forwardFunc.name}`,
      inputTypes,
      outputTypes
    );

    const builder = new IRBuilder(jointFunc);
    const jointArgs = jointFunc.args;

    const fwdInputArgs = jointArgs.slice(0, forwardFunc.inputTypes.length);
    const gradOutputArgs = jointArgs.slice(forwardFunc.inputTypes.length);

    const valueMap = new Map();
    for (let i = 0; i < forwardInputs.length; i++) {
      valueMap.set(forwardInputs[i].id, fwdInputArgs[i]);
    }

    for (const op of topoOrder) {
      if (op.opName === 'return') continue;

      const newOperands = new Array(op.numOperands);
      for (let o = 0; o < op.numOperands; o++) {
        const origVal = op.getOperand(o);
        newOperands[o] = valueMap.get(origVal.id) || origVal;
      }

      const resultTypes = op.results.map(r => r.type);
      const cloned = builder._buildOp(op.opName, newOperands, resultTypes, new Map(op.attributes), null);

      for (let r = 0; r < op.numResults; r++) {
        valueMap.set(op.getResult(r).id, cloned.getResult(r));
      }
    }

    const fwdOutputValues = forwardOutputs.map(v => valueMap.get(v.id));

    const needsGrad = this._computeGradReachability(forwardFunc, topoOrder);
    const segments = this._checkpointPolicy.segment(topoOrder, forwardFunc);
    const accumulator = new GradAccumulator(builder);

    for (let i = 0; i < forwardOutputs.length; i++) {
      accumulator.accumulate(forwardOutputs[i].id, gradOutputArgs[i]);
    }

    for (let s = segments.length - 1; s >= 0; s--) {
      const seg = segments[s];

      const recomputeMap = new Map();

      for (const op of seg.ops) {
        const newOperands = new Array(op.numOperands);
        for (let o = 0; o < op.numOperands; o++) {
          const origVal = op.getOperand(o);
          newOperands[o] = recomputeMap.get(origVal.id) ||
                           valueMap.get(origVal.id) ||
                           origVal;
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
          const origVal = op.getOperand(o);
          operandValues[o] = recomputeMap.get(origVal.id) ||
                             valueMap.get(origVal.id) ||
                             origVal;
        }

        const resultValues = new Array(op.numResults);
        for (let r = 0; r < op.numResults; r++) {
          const origRes = op.getResult(r);
          resultValues[r] = recomputeMap.get(origRes.id) ||
                            valueMap.get(origRes.id) ||
                            origRes;
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

    const gradInputValues = [];
    for (let i = 0; i < forwardInputs.length; i++) {
      const grad = accumulator.get(forwardInputs[i].id);
      if (grad) {
        gradInputValues.push(grad);
      } else {
        const zeroConst = builder.scalarConstant(0, forwardInputs[i].type.dtype).getResult(0);
        const zeroBroadcast = builder.broadcast(zeroConst, forwardInputs[i].type.shape, []).getResult(0);
        gradInputValues.push(zeroBroadcast);
      }
    }

    builder.returnOp([...fwdOutputValues, ...gradInputValues]);

    return {
      jointFunc,
      numForwardOutputs: forwardOutputs.length,
      numGradInputs: gradInputValues.length,
    };
  }
}
