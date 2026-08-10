import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { UseDefAnalysis } from '../analysis/use_def.js';
import { GradAccumulator, gradOrZero } from './grad_accumulator.js';
import { getVJPRule, isGradientBarrier, requireVJPRuleOrBarrier } from './vjp_registry.js';
import { RematPolicy } from './remat_policy.js';
import { REGION_CONTROL_FLOW, backpropOps } from './backward_builder.js';

function replayOp(builder, op, resolve, record) {
  const vmap = new Map();
  for (const operand of op.operands) vmap.set(operand, resolve(operand));
  const cloned = op.clone(vmap);
  builder._insert(cloned);
  for (let r = 0; r < op.numResults; r++) record(op.getResult(r), cloned.getResult(r));
  return cloned;
}

export class JointGraphBuilder {
  constructor(opts = {}) {
    this._rematPolicy = opts.rematPolicy || new RematPolicy(opts.remat || {});
    this._checkpointPolicy = opts.checkpointPolicy || null;
  }

  build(forwardFunc) {
    if (this._checkpointPolicy) {
      return this._buildCheckpointed(forwardFunc);
    }
    const s = this._buildScaffold(forwardFunc);
    backpropOps(s.topoOrder, {
      accumulator: s.accumulator, builder: s.builder, needsGrad: s.needsGrad,
      resolveValue: (v) => s.valueMap.get(v.id) || v,
    });
    return this._finish(s);
  }

  _buildCheckpointed(forwardFunc) {
    const s = this._buildScaffold(forwardFunc);
    const segments = this._checkpointPolicy.segment(s.topoOrder, forwardFunc);

    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      const recomputeMap = new Map();

      for (const op of seg.ops) {
        replayOp(
          s.builder, op,
          (v) => recomputeMap.get(v.id) || s.valueMap.get(v.id) || v,
          (orig, next) => recomputeMap.set(orig.id, next),
        );
      }

      backpropOps(seg.ops, {
        accumulator: s.accumulator, builder: s.builder, needsGrad: s.needsGrad,
        resolveValue: (v) => recomputeMap.get(v.id) || s.valueMap.get(v.id) || v,
      });
    }
    return this._finish(s);
  }

  _buildScaffold(forwardFunc) {
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;
    this._assertNoRegionControlFlow(topoOrder);

    const returnOp = forwardFunc.getReturnOp();
    if (!returnOp) throw new Error('Forward function has no return op');

    const forwardOutputs = returnOp.operands;
    const forwardInputs = forwardFunc.args;

    const gradOutputTypes = forwardOutputs.map(v => v.type);
    const inputTypes = [...forwardFunc.inputTypes, ...gradOutputTypes];
    const outputTypes = [...forwardFunc.outputTypes, ...forwardFunc.inputTypes];

    const jointFunc = new GraphFunction(`joint_${forwardFunc.name}`, inputTypes, outputTypes);
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
      replayOp(
        builder, op,
        (v) => valueMap.get(v.id) || v,
        (orig, next) => valueMap.set(orig.id, next),
      );
    }

    const fwdOutputValues = forwardOutputs.map(v => valueMap.get(v.id));
    const needsGrad = this._computeGradReachability(forwardFunc, topoOrder);
    const accumulator = new GradAccumulator(builder);
    for (let i = 0; i < forwardOutputs.length; i++) {
      accumulator.accumulate(forwardOutputs[i].id, gradOutputArgs[i]);
    }

    return { topoOrder, forwardInputs, forwardOutputs, fwdOutputValues, valueMap, builder, needsGrad, accumulator, jointFunc };
  }

  _finish(s) {
    const gradInputValues = [];
    for (let i = 0; i < s.forwardInputs.length; i++) {
      gradInputValues.push(gradOrZero(s.builder, s.forwardInputs[i], s.accumulator));
    }
    s.builder.returnOp([...s.fwdOutputValues, ...gradInputValues]);
    return {
      jointFunc: s.jointFunc,
      numForwardOutputs: s.forwardOutputs.length,
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
      if (isGradientBarrier(op.opName)) continue;

      for (let o = 0; o < op.numOperands; o++) {
        needsGrad.add(op.getOperand(o).id);
      }
    }

    return needsGrad;
  }

  _assertNoRegionControlFlow(ops) {
    for (const op of ops) {
      if (REGION_CONTROL_FLOW.has(op.opName)) {
        throw new Error(`JointGraphBuilder does not support region control-flow op '${op.opName}'; use BackwardGraphBuilder (separate mode) without a checkpointPolicy, which differentiates scan/if.`);
      }
    }
  }
}
