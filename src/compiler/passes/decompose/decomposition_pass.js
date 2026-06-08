import { FunctionPass, PassResult } from '../pass.js';
import { IRBuilder, broadcastDimsExcluding } from '../../ir/graph/builder.js';
import { ScalarType, TensorType } from '../../ir/graph/types.js';
import { TraceLevel } from '../../pipeline/trace.js';

const decompositionRules = new Map();

export function registerDecomposition(opName, ruleFn) {
  decompositionRules.set(opName, ruleFn);
}

export function hasDecomposition(opName) {
  return decompositionRules.has(opName);
}

export class DecompositionPass extends FunctionPass {
  constructor() {
    super('DecompositionPass');
  }

  run(func) {
    const worklist = [];
    for (const op of func.ops()) {
      if (decompositionRules.has(op.opName)) worklist.push(op);
    }
    if (worklist.length === 0) return PassResult.UNCHANGED;

    const builder = new IRBuilder(func);
    const decomposed = [];

    for (const op of worklist) {
      if (!op.parentBlock) continue;
      const rule = decompositionRules.get(op.opName);
      builder.block = op.parentBlock;
      builder.setInsertionPoint(op);
      decomposed.push(op.opName);
      rule(op, builder);
    }

    if (this.trace && this.trace.level >= TraceLevel.DEBUG) {
      const counts = {};
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

registerDecomposition('softmax', (op, b) => {
  const input = op.getOperand(0);
  const axis = op.getAttr('axis');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
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
  const axis = op.getAttr('axis');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
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

registerDecomposition('sigmoid', (op, b) => {
  const x = op.getOperand(0);
  const dtype = x.type.dtype;
  const shape = x.type.shape;

  const negX = b.neg(x);
  const expNeg = b.exp(negX.getResult(0));
  const one = b.broadcast(b.scalarConstant(1, dtype).getResult(0), shape, []);
  const denom = b.add(one.getResult(0), expNeg.getResult(0));
  const result = b.div(one.getResult(0), denom.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('gelu', (op, b) => {
  const x = op.getOperand(0);
  const dtype = x.type.dtype;
  const shape = x.type.shape;

  const coeff = b.broadcast(b.scalarConstant(1.702, dtype).getResult(0), shape, []);
  const scaled = b.mul(coeff.getResult(0), x);
  const negScaled = b.neg(scaled.getResult(0));
  const expNeg = b.exp(negScaled.getResult(0));
  const one = b.broadcast(b.scalarConstant(1, dtype).getResult(0), shape, []);
  const denom = b.add(one.getResult(0), expNeg.getResult(0));
  const sig = b.div(one.getResult(0), denom.getResult(0));
  const result = b.mul(x, sig.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('silu', (op, b) => {
  const x = op.getOperand(0);
  const dtype = x.type.dtype;
  const shape = x.type.shape;

  const negX = b.neg(x);
  const expNeg = b.exp(negX.getResult(0));
  const one = b.broadcast(b.scalarConstant(1, dtype).getResult(0), shape, []);
  const denom = b.add(one.getResult(0), expNeg.getResult(0));
  const sig = b.div(one.getResult(0), denom.getResult(0));
  const result = b.mul(x, sig.getResult(0));

  op.replaceAllResultsWith([result.getResult(0)]);
  op.erase();
});

registerDecomposition('layer_norm', (op, b) => {
  const input = op.getOperand(0);
  const gamma = op.getOperand(1);
  const beta = op.getOperand(2);
  const axis = op.getAttr('axis');
  const eps = op.getAttr('epsilon');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const bcastDims = broadcastDimsExcluding(rank, axis);

  const meanVal = b.reduce(input, b.scalarConstant(0, dtype).getResult(0), [axis], 'mean');
  const bcastMean = b.broadcast(meanVal.getResult(0), shape, bcastDims);
  const centered = b.sub(input, bcastMean.getResult(0));
  const sq = b.mul(centered.getResult(0), centered.getResult(0));
  const variance = b.reduce(sq.getResult(0), b.scalarConstant(0, dtype).getResult(0), [axis], 'mean');
  const epsConst = b.broadcast(b.scalarConstant(eps, dtype).getResult(0), variance.getResult(0).type.shape, []);
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
  const axis = op.getAttr('axis');
  const eps = op.getAttr('epsilon');
  const rank = input.type.rank;
  const dtype = input.type.dtype;
  const shape = input.type.shape;

  const epsConst = b.broadcast(b.scalarConstant(eps, dtype).getResult(0), variance.type.shape, []);
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
  if (cond.type.dtype !== ScalarType.BOOL) {
    const zero = b.broadcast(b.scalarConstant(0, cond.type.dtype).getResult(0), cond.type.shape, []);
    cond = b.compare(cond, zero.getResult(0), 'ne').getResult(0);
  }
  const sel = b.select(cond, op.getOperand(1), op.getOperand(2));
  op.replaceAllResultsWith([sel.getResult(0)]);
  op.erase();
});

registerDecomposition('split', (op, b) => {
  const input = op.getOperand(0);
  const dim = op.getAttr('dimension');
  const sizes = op.getAttr('split_sizes');
  const shape = input.type.shape;
  const results = [];
  let offset = 0;
  for (const size of sizes) {
    const starts = shape.map((_, i) => i === dim ? offset : 0);
    const limits = shape.map((s, i) => i === dim ? offset + size : s);
    results.push(b.slice(input, starts, limits).getResult(0));
    offset += size;
  }
  op.replaceAllResultsWith(results);
  op.erase();
});

registerDecomposition('one_hot', (op, b) => {
  const indices = op.getOperand(0);
  const depth = op.getAttr('depth');
  const axis = op.getAttr('axis') ?? -1;
  const onVal = op.getAttr('on_value') ?? 1;
  const offVal = op.getAttr('off_value') ?? 0;
  const resultType = op.getResult(0).type;
  const dtype = resultType.dtype;
  const resultShape = resultType.shape;
  const resolvedAxis = axis < 0 ? indices.type.rank + 1 + axis : axis;
  const iotaType = new TensorType(resultShape, ScalarType.I32);
  const iotaOp = b._inferAndBuild('iota', [], { iota_dimension: resolvedAxis, tensor_type: iotaType });
  const expandedShape = [...indices.type.shape];
  expandedShape.splice(resolvedAxis, 0, 1);
  const reshaped = b.reshape(indices, expandedShape);
  const bcastDims = Array.from({length: expandedShape.length}, (_, i) => i);
  const bcastIndices = b.broadcast(reshaped.getResult(0), resultShape, bcastDims);
  const converted = b.convert(bcastIndices.getResult(0), ScalarType.I32);
  const cmp = b.compare(converted.getResult(0), iotaOp.getResult(0), 'eq');
  const onBcast = b.broadcast(b.scalarConstant(onVal, dtype).getResult(0), resultShape, []);
  const offBcast = b.broadcast(b.scalarConstant(offVal, dtype).getResult(0), resultShape, []);
  const sel = b.select(cmp.getResult(0), onBcast.getResult(0), offBcast.getResult(0));
  op.replaceAllResultsWith([sel.getResult(0)]);
  op.erase();
});

function bcast(b, scalar, dtype, shape) {
  return b.broadcast(b.scalarConstant(scalar, dtype).getResult(0), shape, []).getResult(0);
}

registerDecomposition('elu', (op, b) => {
  const x = op.getOperand(0);
  const alpha = op.getAttr('alpha') ?? 1.0;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const slope = op.getAttr('negative_slope') ?? 0.01;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const alpha = op.getAttr('alpha') ?? 1.0;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const dtype = x.type.dtype;
  const shape = x.type.shape;
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
  const D = weight.type.shape[weight.type.rank - 1];
  const idxRank = indices.type.rank;
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
