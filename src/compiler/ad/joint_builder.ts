import { GraphFunction } from '../ir/graph/function.js';
import { IRBuilder } from '../ir/graph/builder.js';
import { UseDefAnalysis } from '../analysis/use_def.js';
import { readValues } from '../ir/graph/graph_algorithms.js';
import { GradAccumulator, gradOrZero } from './grad_accumulator.js';
import { getRegionVJP } from './vjp_registry.js';
import { RematPolicy } from './remat_policy.js';
import { backpropOps, computeGradReachability } from './backward_builder.js';
import type { Operation } from '../ir/graph/operation.js';
import type { Value } from '../ir/graph/value.js';
import type { CheckpointPolicy } from './checkpoint_policy.js';
import type { RematPolicyOpts } from './remat_policy.js';

export type JointBuilderOpts = Readonly<{
  rematPolicy?: RematPolicy;
  remat?: RematPolicyOpts;
  checkpointPolicy?: CheckpointPolicy | null;
}>;

export type JointBuildResult = {
  jointFunc: GraphFunction;
  numForwardOutputs: number;
  numGradInputs: number;
};

type JointScaffold = {
  topoOrder: readonly Operation[];
  forwardInputs: readonly Value[];
  forwardOutputs: readonly Value[];
  fwdOutputValues: Value[];
  valueMap: Map<number, Value>;
  builder: IRBuilder;
  needsGrad: Set<number>;
  accumulator: GradAccumulator;
  jointFunc: GraphFunction;
};

type ResolveFn = (v: Value) => Value;
type RecordFn = (orig: Value, next: Value) => void;

function replayOp(builder: IRBuilder, op: Operation, resolve: ResolveFn, record: RecordFn): Operation {
  const vmap = new Map<Value, Value>();
  for (const value of readValues(op)) vmap.set(value, resolve(value));
  const cloned = op.clone(vmap);
  builder._insert(cloned);
  for (let r = 0; r < op.numResults; r++) record(op.getResult(r), cloned.getResult(r));
  return cloned;
}

function regionBackprop(accumulator: GradAccumulator, builder: IRBuilder, needsGrad: ReadonlySet<number>, resolve: ResolveFn) {
  return (op: Operation): boolean => {
    const regionFn = getRegionVJP(op.opName);
    if (!regionFn) return false;
    (regionFn as unknown as (op: Operation, ctx: unknown) => void)(op, {
      accumulator, builder, needsGrad, materialize: resolve,
    });
    return true;
  };
}

export class JointGraphBuilder {
  private _rematPolicy: RematPolicy;
  private _checkpointPolicy: CheckpointPolicy | null;

  constructor(opts: JointBuilderOpts = {}) {
    this._rematPolicy = opts.rematPolicy || new RematPolicy(opts.remat || {});
    this._checkpointPolicy = opts.checkpointPolicy || null;
  }

  build(forwardFunc: GraphFunction): JointBuildResult {
    if (this._checkpointPolicy) {
      return this._buildCheckpointed(forwardFunc);
    }
    const s = this._buildScaffold(forwardFunc);
    const resolve = (v: Value) => s.valueMap.get(v.id) || v;
    backpropOps(s.topoOrder, {
      accumulator: s.accumulator, builder: s.builder, needsGrad: s.needsGrad,
      resolveValue: resolve,
      handleRegionOp: regionBackprop(s.accumulator, s.builder, s.needsGrad, resolve),
    });
    return this._finish(s);
  }

  _buildCheckpointed(forwardFunc: GraphFunction): JointBuildResult {
    const s = this._buildScaffold(forwardFunc);
    const segments = (this._checkpointPolicy as CheckpointPolicy).segment(s.topoOrder, forwardFunc);

    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      const recomputeMap = new Map<number, Value>();
      const resolve = (v: Value) => recomputeMap.get(v.id) || s.valueMap.get(v.id) || v;

      for (const op of seg.ops) {
        replayOp(s.builder, op, resolve, (orig, next) => recomputeMap.set(orig.id, next));
      }

      backpropOps(seg.ops, {
        accumulator: s.accumulator, builder: s.builder, needsGrad: s.needsGrad,
        resolveValue: resolve,
        handleRegionOp: regionBackprop(s.accumulator, s.builder, s.needsGrad, resolve),
      });
    }
    return this._finish(s);
  }

  _buildScaffold(forwardFunc: GraphFunction): JointScaffold {
    const analysis = UseDefAnalysis.compute(forwardFunc);
    const topoOrder = analysis.topologicalOrder;

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

    const valueMap = new Map<number, Value>();
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

    const fwdOutputValues = forwardOutputs.map(v => valueMap.get(v.id) as Value);
    const needsGrad = computeGradReachability(forwardFunc, topoOrder);
    const accumulator = new GradAccumulator(builder);
    for (let i = 0; i < forwardOutputs.length; i++) {
      accumulator.accumulate(forwardOutputs[i].id, gradOutputArgs[i]);
    }

    return { topoOrder, forwardInputs, forwardOutputs, fwdOutputValues, valueMap, builder, needsGrad, accumulator, jointFunc };
  }

  _finish(s: JointScaffold): JointBuildResult {
    const gradInputValues: Value[] = [];
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
}
