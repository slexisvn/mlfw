import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder, broadcastDimsExcluding } from '../../ir/graph/builder.js';
import { ScalarType, TensorType } from '../../ir/graph/types.js';
import { TraceLevel } from '../../support/trace.js';
import { explainer } from '../explain.js';
import type { GraphFunction } from '../../ir/graph/function.js';
import type { Operation } from '../../ir/graph/operation.js';
import type { Value } from '../../ir/graph/value.js';
import type { Block } from '../../ir/graph/block.js';
import type { ScalarDType, Shape } from '../../ir/graph/types.js';
import type { PassResultValue, PassTarget } from '../pass.js';
import type { CompileTarget } from '../../support/config_types.js';

export type DecompositionRule = (op: Operation, b: IRBuilder) => void;

const decompositionRules = new Map<string, DecompositionRule>();

export function registerDecomposition(opName: string, ruleFn: DecompositionRule): void {
  decompositionRules.set(opName, ruleFn);
}

export function unregisterDecomposition(opName: string): boolean {
  return decompositionRules.delete(opName);
}

export function hasDecomposition(opName: string): boolean {
  return decompositionRules.has(opName);
}

export class DecompositionPass extends FunctionPass {
  target: CompileTarget | null;

  constructor(target: CompileTarget | null = null) {
    super('DecompositionPass');
    this.target = target;
  }

  _shouldDecompose(op: Operation): boolean {
    if (!this.target) return true;
    const native = this.target.getAttr ? this.target.getAttr<ReadonlySet<string>>('nativeOps') : null;
    return !(native && native.has(op.opName));
  }

  override run(func: PassTarget): PassResultValue {
    const worklist: Operation[] = [];
    for (const op of (func as GraphFunction).opsRecursive()) {
      if (decompositionRules.has(op.opName) && this._shouldDecompose(op)) worklist.push(op);
    }
    if (worklist.length === 0) return PassResult.UNCHANGED;

    const builder = new IRBuilder(func as GraphFunction);
    const explain = explainer(this.trace, this.name);
    const decomposed: string[] = [];

    for (const op of worklist) {
      if (!op.parentBlock) continue;
      const rule = decompositionRules.get(op.opName) as DecompositionRule;
      builder.block = op.parentBlock as Block;
      builder.setInsertionPoint(op);
      decomposed.push(op.opName);
      const sizeBefore = (op.parentBlock as Block).size;
      builder.withLocation(op.loc, () => rule(op, builder));
      if (explain) {
        explain(op.opName, 'rewritten into primitives',
          'no lowering rule exists for this op, so it is re-expressed with ops every target can lower',
          { opsAdded: (op.parentBlock as Block | null) ? (op.parentBlock as Block).size - sizeBefore : 0 });
      }
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      const counts: Record<string, number> = {};
      for (const name of decomposed) counts[name] = (counts[name] || 0) + 1;
      this.trace.emit({
        type: 'pass_detail', passName: this.name,
        decomposed: counts, totalDecomposed: decomposed.length,
        level: TraceLevel.DEBUG,
      });
    }

    return PassResult.CHANGED;
  }
}

registerDecomposition('stop_gradient', (op) => {
  op.replaceAllResultsWith([op.getOperand(0)]);
  op.erase();
});

registerDecomposition('all_reduce', (op, b) => {
  const x = op.getOperand(0);
  const axis = op.getAttr<number>('mesh_axis') ?? 0;
  const reduceOp = op.getAttr<string>('reduce_op') || 'sum';
  const shape = (x.type as TensorType).shape;
  const dtype = (x.type as TensorType).dtype;
  const init = reduceOp === 'max' ? -Infinity
    : reduceOp === 'min' ? Infinity
    : (reduceOp === 'prod' || reduceOp === 'and') ? 1
    : 0;
  const red = b.reduce(x, b.scalarConstant(init, dtype).getResult(0), [axis], reduceOp).getResult(0);
  const bcastDims: number[] = [];
  for (let i = 0; i < shape.length; i++) if (i !== axis) bcastDims.push(i);
  const out = b.broadcast(red, shape, bcastDims).getResult(0);
  op.replaceAllResultsWith([out]);
  op.erase();
});

registerDecomposition('all_gather', (op, b) => {
  const x = op.getOperand(0);
  const shape = (x.type as TensorType).shape;
  const meshAxis = op.getAttr<number>('mesh_axis') ?? 0;
  const gatherDim = op.getAttr<number>('gather_dim') ?? 1;
  const Ndev = shape[meshAxis] as number;
  const squeezed = shape.filter((_, i) => i !== meshAxis) as number[];
  const gatherInSqueezed = gatherDim < meshAxis ? gatherDim : gatherDim - 1;
  const shards: Value[] = [];
  for (let d = 0; d < Ndev; d++) {
    const starts: number[] = shape.map((_, i) => (i === meshAxis ? d : 0));
    const limits: number[] = shape.map((dim, i) => (i === meshAxis ? d + 1 : dim) as number);
    const sliced = b.slice(x, starts, limits).getResult(0);
    shards.push(b.reshape(sliced, squeezed).getResult(0));
  }
  const gathered = shards.length === 1 ? shards[0] : b.concat(shards, gatherInSqueezed).getResult(0);
  const outShape: number[] = [...shape] as number[];
  outShape[gatherDim] = Ndev * (shape[gatherDim] as number);
  const bcastDims: number[] = [];
  for (let i = 0; i < outShape.length; i++) if (i !== meshAxis) bcastDims.push(i);
  const out = b.broadcast(gathered, outShape, bcastDims).getResult(0);
  op.replaceAllResultsWith([out]);
  op.erase();
});

registerDecomposition('softmax', (op, b) => {
  const input = op.getOperand(0);
  const axis = op.getAttr<number>('axis') as number;
  const dtype = (input.type as TensorType).dtype;
  const rank = (input.type as TensorType).rank;
  const shape = (input.type as TensorType).shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const maxVal = b.reduce(input, b.scalarConstant(-Infinity, dtype).getResult(0), [axis], 'max');
  const bcastMax = b.broadcast(maxVal.getResult(0), shape, bcastDims);
  const shifted = b.sub(input, bcastMax.getResult(0));
  const exps = b.exp(shifted.getResult(0));
  const sumVal = b.reduce(exps.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'sum');
  const bcastSum = b.broadcast(sumVal.getResult(0), shape, bcastDims);
  const result = b.div(exps.getResult(0), bcastSum.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('log_softmax', (op, b) => {
  const input = op.getOperand(0);
  const axis = op.getAttr<number>('axis') as number;
  const rank = (input.type as TensorType).rank;
  const dtype = (input.type as TensorType).dtype;
  const shape = (input.type as TensorType).shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const maxVal = b.reduce(input, b.scalarConstant(-Infinity, dtype).getResult(0), [axis], 'max');
  const bcastMax = b.broadcast(maxVal.getResult(0), shape, bcastDims);
  const shifted = b.sub(input, bcastMax.getResult(0));
  const exps = b.exp(shifted.getResult(0));
  const sumVal = b.reduce(exps.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'sum');
  const logSum = b.log(sumVal.getResult(0));
  const bcastLogSum = b.broadcast(logSum.getResult(0), shape, bcastDims);
  const result = b.sub(shifted.getResult(0), bcastLogSum.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

function emitSigmoid(b: IRBuilder, x: Value, dtype: string, shape: Shape): Value {
  const negX = b.neg(x).getResult(0);
  const expNeg = b.exp(negX).getResult(0);
  const one = b.broadcast(b.scalarConstant(1, dtype as ScalarDType).getResult(0), shape, []).getResult(0);
  const denom = b.add(one, expNeg).getResult(0);
  return b.div(one, denom).getResult(0);
}

registerDecomposition('sigmoid', (op, b) => {
  const x = op.getOperand(0);
  const result = emitSigmoid(b, x, (x.type as TensorType).dtype, (x.type as TensorType).shape);
  op.replaceAllResultsWith([result]);
  op.erase();
});

registerDecomposition('gelu', (op, b) => {
  const x = op.getOperand(0);
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const coeff = b.broadcast(b.scalarConstant(1.702, dtype).getResult(0), shape, []).getResult(0);
  const scaled = b.mul(coeff, x).getResult(0);
  const sig = emitSigmoid(b, scaled, dtype, shape);
  const result = b.mul(x, sig).getResult(0);
  op.replaceAllResultsWith([result]);
  op.erase();
});

registerDecomposition('silu', (op, b) => {
  const x = op.getOperand(0);
  const sig = emitSigmoid(b, x, (x.type as TensorType).dtype, (x.type as TensorType).shape);
  const result = b.mul(x, sig).getResult(0);
  op.replaceAllResultsWith([result]);
  op.erase();
});

registerDecomposition('layer_norm', (op, b) => {
  const input = op.getOperand(0);
  const gamma = op.getOperand(1);
  const beta = op.getOperand(2);
  const axis = op.getAttr<number>('axis') as number;
  const eps = op.getAttr<number>('epsilon') as number;
  const rank = (input.type as TensorType).rank;
  const dtype = (input.type as TensorType).dtype;
  const shape = (input.type as TensorType).shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const meanVal = b.reduce(input, b.scalarConstant(0, dtype).getResult(0), [axis], 'mean');
  const bcastMean = b.broadcast(meanVal.getResult(0), shape, bcastDims);
  const centered = b.sub(input, bcastMean.getResult(0));
  const sq = b.mul(centered.getResult(0), centered.getResult(0));
  const variance = b.reduce(sq.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'mean');
  const epsConst = b.broadcast(b.scalarConstant(eps, dtype).getResult(0), (variance.getResult(0).type as TensorType).shape, []);
  const varPlusEps = b.add(variance.getResult(0), epsConst.getResult(0));
  const rstd = b.rsqrt(varPlusEps.getResult(0));
  const bcastRstd = b.broadcast(rstd.getResult(0), shape, bcastDims);
  const normalized = b.mul(centered.getResult(0), bcastRstd.getResult(0));
  const bcastGamma = b.broadcast(gamma, shape, [axis]);
  const scaled = b.mul(normalized.getResult(0), bcastGamma.getResult(0));
  const bcastBeta = b.broadcast(beta, shape, [axis]);
  const result = b.add(scaled.getResult(0), bcastBeta.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('batch_norm', (op, b) => {
  const input = op.getOperand(0);
  const gamma = op.getOperand(1);
  const beta = op.getOperand(2);
  const mean = op.getOperand(3);
  const variance = op.getOperand(4);
  const axis = op.getAttr<number>('axis') as number;
  const eps = op.getAttr<number>('epsilon') as number;
  const dtype = (input.type as TensorType).dtype;
  const shape = (input.type as TensorType).shape;

  const epsConst = b.broadcast(b.scalarConstant(eps, dtype).getResult(0), (variance.type as TensorType).shape, []);
  const varPlusEps = b.add(variance, epsConst.getResult(0));
  const rstd = b.rsqrt(varPlusEps.getResult(0));
  const bcastMean = b.broadcast(mean, shape, [axis]);
  const centered = b.sub(input, bcastMean.getResult(0));
  const bcastRstd = b.broadcast(rstd.getResult(0), shape, [axis]);
  const normalized = b.mul(centered.getResult(0), bcastRstd.getResult(0));
  const bcastGamma = b.broadcast(gamma, shape, [axis]);
  const scaled = b.mul(normalized.getResult(0), bcastGamma.getResult(0));
  const bcastBeta = b.broadcast(beta, shape, [axis]);
  const result = b.add(scaled.getResult(0), bcastBeta.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('where', (op, b) => {
  let cond = op.getOperand(0);
  if ((cond.type as TensorType).dtype !== ScalarType.BOOL) {
    const zero = b.broadcast(b.scalarConstant(0, (cond.type as TensorType).dtype).getResult(0), (cond.type as TensorType).shape, []);
    cond = b.compare(cond, zero.getResult(0), 'ne').getResult(0);
  }
  const sel = b.select(cond, op.getOperand(1), op.getOperand(2));
  op.replaceAllResultsWith([sel.getResult(0)]);
  op.erase();
});

registerDecomposition('split', (op, b) => {
  const input = op.getOperand(0);
  const dim = op.getAttr<number>('dimension') as number;
  const sizes = op.getAttr<readonly number[]>('split_sizes') as readonly number[];
  const shape = (input.type as TensorType).shape;
  const results: Value[] = [];
  let offset = 0;
  for (const size of sizes) {
    const starts: number[] = shape.map((_, i) => i === dim ? offset : 0);
    const limits: number[] = shape.map((s, i) => (i === dim ? offset + size : s) as number);
    results.push(b.slice(input, starts, limits).getResult(0));
    offset += size;
  }
  op.replaceAllResultsWith(results);
  op.erase();
});

registerDecomposition('one_hot', (op, b) => {
  const indices = op.getOperand(0);
  const axis = op.getAttr<number>('axis') ?? -1;
  const onVal = op.getAttr<number>('on_value') ?? 1;
  const offVal = op.getAttr<number>('off_value') ?? 0;
  const resultType = op.getResult(0).type as TensorType;
  const dtype = resultType.dtype;
  const resultShape = resultType.shape;
  const resolvedAxis = axis < 0 ? (indices.type as TensorType).rank + 1 + axis : axis;
  const iotaType = new TensorType(resultShape, ScalarType.I32);
  const iotaOp = b._inferAndBuild('iota', [], { iota_dimension: resolvedAxis, tensor_type: iotaType });
  const idxRank = (indices.type as TensorType).rank;
  const bcastDims = [];
  for (let i = 0; i < idxRank; i++) bcastDims.push(i < resolvedAxis ? i : i + 1);
  const bcastIndices = b.broadcast(indices, resultShape, bcastDims);
  const converted = b.convert(bcastIndices.getResult(0), ScalarType.I32);
  const cmp = b.compare(converted.getResult(0), iotaOp.getResult(0), 'eq');
  const onBcast = b.broadcast(b.scalarConstant(onVal, dtype).getResult(0), resultShape, []);
  const offBcast = b.broadcast(b.scalarConstant(offVal, dtype).getResult(0), resultShape, []);
  const sel = b.select(cmp.getResult(0), onBcast.getResult(0), offBcast.getResult(0));
  op.replaceAllResultsWith([sel.getResult(0)]);
  op.erase();
});

function bcast(b: IRBuilder, scalar: number, dtype: string, shape: Shape): Value {
  return b.broadcast(b.scalarConstant(scalar, dtype as ScalarDType).getResult(0), shape, []).getResult(0);
}

registerDecomposition('elu', (op, b) => {
  const x = op.getOperand(0);
  const alpha = op.getAttr<number>('alpha') ?? 1.0;
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const zero = bcast(b, 0, dtype, shape);
  const mask = b.compare(x, zero, 'gt').getResult(0);
  const one = bcast(b, 1, dtype, shape);
  const expX = b.exp(x).getResult(0);
  const expMinusOne = b.sub(expX, one).getResult(0);
  const alphaVal = bcast(b, alpha, dtype, shape);
  const negBranch = b.mul(alphaVal, expMinusOne).getResult(0);
  const result = b.select(mask, x, negBranch);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('leaky_relu', (op, b) => {
  const x = op.getOperand(0);
  const slope = op.getAttr<number>('negative_slope') ?? 0.01;
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const zero = bcast(b, 0, dtype, shape);
  const mask = b.compare(x, zero, 'gt').getResult(0);
  const slopeVal = bcast(b, slope, dtype, shape);
  const negBranch = b.mul(slopeVal, x).getResult(0);
  const result = b.select(mask, x, negBranch);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('celu', (op, b) => {
  const x = op.getOperand(0);
  const alpha = op.getAttr<number>('alpha') ?? 1.0;
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const zero = bcast(b, 0, dtype, shape);
  const positivePart = b.maximum(x, zero).getResult(0);
  const alphaVal = bcast(b, alpha, dtype, shape);
  const xOverAlpha = b.div(x, alphaVal).getResult(0);
  const expTerm = b.exp(xOverAlpha).getResult(0);
  const one = bcast(b, 1, dtype, shape);
  const expMinusOne = b.sub(expTerm, one).getResult(0);
  const scaled = b.mul(alphaVal, expMinusOne).getResult(0);
  const negativePart = b.minimum(zero, scaled).getResult(0);
  const result = b.add(positivePart, negativePart);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('selu', (op, b) => {
  const x = op.getOperand(0);
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const lambda = 1.0507009873554805;
  const alphaConst = 1.6732632423543772;
  const zero = bcast(b, 0, dtype, shape);
  const mask = b.compare(x, zero, 'gt').getResult(0);
  const one = bcast(b, 1, dtype, shape);
  const expX = b.exp(x).getResult(0);
  const expMinusOne = b.sub(expX, one).getResult(0);
  const alphaVal = bcast(b, alphaConst, dtype, shape);
  const negBranch = b.mul(alphaVal, expMinusOne).getResult(0);
  const inner = b.select(mask, x, negBranch).getResult(0);
  const lambdaVal = bcast(b, lambda, dtype, shape);
  const result = b.mul(lambdaVal, inner);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('mish', (op, b) => {
  const x = op.getOperand(0);
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const one = bcast(b, 1, dtype, shape);
  const expX = b.exp(x).getResult(0);
  const onePlusExp = b.add(one, expX).getResult(0);
  const softplus = b.log(onePlusExp).getResult(0);
  const tanhSoftplus = b.tanh(softplus).getResult(0);
  const result = b.mul(x, tanhSoftplus);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('hardswish', (op, b) => {
  const x = op.getOperand(0);
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const three = bcast(b, 3, dtype, shape);
  const zero = bcast(b, 0, dtype, shape);
  const six = bcast(b, 6, dtype, shape);
  const xPlus3 = b.add(x, three).getResult(0);
  const clamped = b.minimum(b.maximum(xPlus3, zero).getResult(0), six).getResult(0);
  const scaled = b.div(clamped, six).getResult(0);
  const result = b.mul(x, scaled);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('hardsigmoid', (op, b) => {
  const x = op.getOperand(0);
  const dtype = (x.type as TensorType).dtype;
  const shape = (x.type as TensorType).shape;
  const six = bcast(b, 6, dtype, shape);
  const half = bcast(b, 0.5, dtype, shape);
  const zero = bcast(b, 0, dtype, shape);
  const one = bcast(b, 1, dtype, shape);
  const xOverSix = b.div(x, six).getResult(0);
  const shifted = b.add(xOverSix, half).getResult(0);
  const result = b.minimum(b.maximum(shifted, zero).getResult(0), one);
  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('embedding', (op, b) => {
  const weight = op.getOperand(0);
  const indices = op.getOperand(1);
  const D = (weight.type as TensorType).shape[(weight.type as TensorType).rank - 1];
  const idxRank = (indices.type as TensorType).rank;
  const gatherOp = b._inferAndBuild('gather', [weight, indices], {
    offset_dims: Array.from({length: 1}, (_, i) => idxRank + i),
    collapsed_slice_dims: [0],
    start_index_map: [0],
    slice_sizes: [1, D],
    index_vector_dim: idxRank,
  });
  op.replaceAllResultsWith([gatherOp.getResult(0)]);
  op.erase();
});
