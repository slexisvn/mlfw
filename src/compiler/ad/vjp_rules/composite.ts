import { registerVJPRule } from '../vjp_registry.js';
import { broadcastDimsExcluding } from '../../ir/graph/builder.js';
import { ScalarType, TensorType } from '../../ir/graph/types.js';
import type { Value } from '../../ir/graph/value.js';

const MASKED_SCORE = -1e30;

registerVJPRule('softmax', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const axis = ctx.op.getAttr<number>('axis')!;
  const dtype = result.type.dtype;
  const shape = result.type.shape;

  const gradTimesSoftmax = ctx.builder.mul(grad, result).getResult(0);
  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const sumGradSoftmax = ctx.builder.reduce(gradTimesSoftmax, initConst, [axis], 'sum').getResult(0);
  const broadcastDims = broadcastDimsExcluding(shape.length, axis);
  const sumBroadcast = ctx.builder.broadcast(sumGradSoftmax, shape, broadcastDims).getResult(0);
  const shifted = ctx.builder.sub(grad, sumBroadcast).getResult(0);
  return [ctx.builder.mul(result, shifted).getResult(0)];
});

registerVJPRule('log_softmax', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const [input] = ctx.operands;
  const axis = ctx.op.getAttr<number>('axis')!;
  const dtype = result.type.dtype;
  const shape = result.type.shape;

  const softmaxVal = ctx.builder.exp(result).getResult(0);
  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const sumGrad = ctx.builder.reduce(grad, initConst, [axis], 'sum').getResult(0);
  const broadcastDims = broadcastDimsExcluding(shape.length, axis);
  const sumBroadcast = ctx.builder.broadcast(sumGrad, shape, broadcastDims).getResult(0);
  const softmaxTimesSum = ctx.builder.mul(softmaxVal, sumBroadcast).getResult(0);
  return [ctx.builder.sub(grad, softmaxTimesSum).getResult(0)];
});

registerVJPRule('layer_norm', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [input, gamma] = ctx.operands;
  const axis = ctx.op.getAttr<number>('axis')!;
  const epsilon = ctx.op.getAttr<number>('epsilon')!;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const normDims = [axis < 0 ? shape.length + axis : axis];
  const normSize = shape[normDims[0]];

  const batchDims = broadcastDimsExcluding(shape.length, normDims[0]);

  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const mean = ctx.builder.reduce(input, initConst, normDims, 'mean').getResult(0);
  const meanBr = ctx.builder.broadcast(mean, shape, batchDims).getResult(0);
  const centered = ctx.builder.sub(input, meanBr).getResult(0);
  const centeredSq = ctx.builder.mul(centered, centered).getResult(0);
  const variance = ctx.builder.reduce(centeredSq, initConst, normDims, 'mean').getResult(0);
  const epsBr = ctx.full(epsilon, variance.type as TensorType);
  const varPlusEps = ctx.builder.add(variance, epsBr).getResult(0);
  const rstd = ctx.builder.rsqrt(varPlusEps).getResult(0);
  const rstdBr = ctx.builder.broadcast(rstd, shape, batchDims).getResult(0);

  const gammaBrDims = [];
  for (let i = 0; i < shape.length; i++) {
    if (normDims.includes(i)) gammaBrDims.push(i);
  }
  const gammaBr = ctx.builder.broadcast(gamma, shape, gammaBrDims).getResult(0);

  const gradTimesGamma = ctx.builder.mul(grad, gammaBr).getResult(0);
  const nBr = ctx.full(normSize as number, input.type as TensorType);

  const term1 = ctx.builder.mul(nBr, gradTimesGamma).getResult(0);
  const sumGradGamma = ctx.builder.reduce(gradTimesGamma, initConst, normDims, 'sum').getResult(0);
  const sumGradGammaBr = ctx.builder.broadcast(sumGradGamma, shape, batchDims).getResult(0);
  const xhat = ctx.builder.mul(centered, rstdBr).getResult(0);
  const gradGammaXhat = ctx.builder.mul(gradTimesGamma, xhat).getResult(0);
  const sumGradGammaXhat = ctx.builder.reduce(gradGammaXhat, initConst, normDims, 'sum').getResult(0);
  const sumGradGammaXhatBr = ctx.builder.broadcast(sumGradGammaXhat, shape, batchDims).getResult(0);
  const term3 = ctx.builder.mul(xhat, sumGradGammaXhatBr).getResult(0);
  const numerator = ctx.builder.sub(term1, ctx.builder.add(sumGradGammaBr, term3).getResult(0)).getResult(0);
  const rstdOverN = ctx.builder.div(rstdBr, nBr).getResult(0);
  const gradInput = ctx.builder.mul(rstdOverN, numerator).getResult(0);

  const gradGamma = ctx.builder.reduce(ctx.builder.mul(grad, xhat).getResult(0), initConst, batchDims, 'sum').getResult(0);
  const gradBeta = ctx.builder.reduce(grad, initConst, batchDims, 'sum').getResult(0);

  return [gradInput, gradGamma, gradBeta];
});

registerVJPRule('scaled_dot_product_attention', (ctx) => {
  const dO = ctx.gradOutputs[0]!;
  const [Q, K, V] = ctx.operands;
  const scale = ctx.op.getAttr('scale');
  const causal = ctx.op.getAttr<boolean>('causal')!;
  const b = ctx.builder;
  const dtype = Q.type.dtype;
  const rank = Q.type.rank;
  const perm: number[] = [];
  for (let i = 0; i < rank; i++) perm.push(i);
  perm[rank - 2] = rank - 1; perm[rank - 1] = rank - 2;
  const lastT = (x: Value) => b.transpose(x, perm).getResult(0);

  const s = b.matmul(Q, lastT(K)).getResult(0);
  let ss = b.mul(s, ctx.full(scale as number, s.type as TensorType)).getResult(0);
  if (causal) {
    const scoreType = ss.type as TensorType;
    const indexType = new TensorType(scoreType.shape, ScalarType.I32);
    const rowIdx = b.iota(rank - 2, indexType).getResult(0);
    const colIdx = b.iota(rank - 1, indexType).getResult(0);
    const offset = (scoreType.shape[rank - 1] as number) - (scoreType.shape[rank - 2] as number);
    const limit = b.add(rowIdx, b.broadcast(b.scalarConstant(offset, ScalarType.I32).getResult(0), scoreType.shape, []).getResult(0)).getResult(0);
    const allowed = b.compare(colIdx, limit, 'le').getResult(0);
    ss = b.select(allowed, ss, ctx.full(MASKED_SCORE, scoreType)).getResult(0);
  }
  const p = b.softmax(ss, rank - 1).getResult(0);

  const dV = b.matmul(lastT(p), dO).getResult(0);
  const dP = b.matmul(dO, lastT(V)).getResult(0);

  const init = b.scalarConstant(0, dtype).getResult(0);
  const dPP = b.mul(dP, p).getResult(0);
  const sumDPP = b.reduce(dPP, init, [rank - 1], 'sum').getResult(0);
  const bcastDims: number[] = [];
  for (let i = 0; i < rank - 1; i++) bcastDims.push(i);
  const sumBr = b.broadcast(sumDPP, (p.type as TensorType).shape, bcastDims).getResult(0);
  const dS = b.mul(p, b.sub(dP, sumBr).getResult(0)).getResult(0);
  const dSraw = b.mul(dS, ctx.full(scale as number, dS.type as TensorType)).getResult(0);

  const dQ = b.matmul(dSraw, K).getResult(0);
  const dK = b.matmul(lastT(dSraw), Q).getResult(0);
  return [dQ, dK, dV];
});

registerVJPRule('pool2d', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [input] = ctx.operands;
  const b = ctx.builder;
  const poolType = ctx.op.getAttr<string>('pool_type')!;
  const ks = ctx.op.getAttr<readonly number[]>('kernel_size')!;
  const strides = ctx.op.getAttr<readonly number[]>('strides')!;
  const padding = ctx.op.getAttr<readonly (readonly number[])[]>('padding')!;
  const layout = ctx.op.getAttr<string>('layout')! || 'NCHW';
  const noPad = padding.every(p => p[0] === 0 && p[1] === 0);
  const strideEqKernel = strides[0] === ks[0] && strides[1] === ks[1];
  if (layout !== 'NCHW' || !noPad || !strideEqKernel || (poolType !== 'avg' && poolType !== 'max')) {
    throw new Error('pool2d VJP supports only non-overlapping (stride=kernel) avg/max pooling without padding, NCHW');
  }

  const [N, C, OH, OW] = grad.type.shape;
  const [kh, kw] = ks;
  const fullShape = input.type.shape;
  const upsample = (v: Value) => b.reshape(b.broadcast(v, [N, C, OH, kh, OW, kw], [0, 1, 2, 4]).getResult(0), fullShape).getResult(0);
  const upGrad = upsample(grad);

  if (poolType === 'avg') {
    const cBr = ctx.full(kh * kw, input.type);
    return [b.div(upGrad, cBr).getResult(0)];
  }
  const upOut = upsample(ctx.results[0]);
  const mask = b.compare(input, upOut, 'eq').getResult(0);
  const zero = ctx.full(0, input.type);
  return [b.select(mask, upGrad, zero).getResult(0)];
});

registerVJPRule('batch_norm', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [input, gamma, , mean, variance] = ctx.operands;
  const axis = ctx.op.getAttr<number>('axis')!;
  const eps = ctx.op.getAttr<number>('epsilon')!;
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const b = ctx.builder;

  const batchDims = broadcastDimsExcluding(shape.length, axis);

  const init = b.scalarConstant(0, dtype).getResult(0);
  const epsConst = ctx.full(eps, variance.type);
  const rstd = b.rsqrt(b.add(variance, epsConst).getResult(0)).getResult(0);
  const rstdBr = b.broadcast(rstd, shape, [axis]).getResult(0);
  const meanBr = b.broadcast(mean, shape, [axis]).getResult(0);
  const gammaBr = b.broadcast(gamma, shape, [axis]).getResult(0);
  const centered = b.sub(input, meanBr).getResult(0);
  const normalized = b.mul(centered, rstdBr).getResult(0);
  const gradGammaBr = b.mul(grad, gammaBr).getResult(0);

  const gradInput = b.mul(gradGammaBr, rstdBr).getResult(0);
  const gradGamma = b.reduce(b.mul(grad, normalized).getResult(0), init, batchDims, 'sum').getResult(0);
  const gradBeta = b.reduce(grad, init, batchDims, 'sum').getResult(0);
  const gradMean = b.neg(b.reduce(gradInput, init, batchDims, 'sum').getResult(0)).getResult(0);

  const rstd3 = b.mul(b.mul(rstdBr, rstdBr).getResult(0), rstdBr).getResult(0);
  const negHalf = ctx.full(-0.5, input.type);
  const gradVarFull = b.mul(b.mul(b.mul(gradGammaBr, centered).getResult(0), rstd3).getResult(0), negHalf).getResult(0);
  const gradVar = b.reduce(gradVarFull, init, batchDims, 'sum').getResult(0);

  return [gradInput, gradGamma, gradBeta, gradMean, gradVar];
});

registerVJPRule('elu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const result = ctx.results[0];
  const alpha = ctx.op.getAttr<number>('alpha')! ?? 1.0;
  const zero = ctx.full(0, x.type);
  const one = ctx.full(1, x.type);
  const alphaVal = ctx.full(alpha, x.type);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const negDeriv = ctx.builder.add(result, alphaVal).getResult(0);
  const deriv = ctx.builder.select(mask, one, negDeriv).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('leaky_relu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const slope = ctx.op.getAttr<number>('negative_slope')! ?? 0.01;
  const zero = ctx.full(0, x.type);
  const one = ctx.full(1, x.type);
  const slopeVal = ctx.full(slope, x.type);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const deriv = ctx.builder.select(mask, one, slopeVal).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('celu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const alpha = ctx.op.getAttr<number>('alpha')! ?? 1.0;
  const zero = ctx.full(0, x.type);
  const one = ctx.full(1, x.type);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const alphaVal = ctx.full(alpha, x.type);
  const xOverAlpha = ctx.builder.div(x, alphaVal).getResult(0);
  const negDeriv = ctx.builder.exp(xOverAlpha).getResult(0);
  const deriv = ctx.builder.select(mask, one, negDeriv).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('selu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const lambda = 1.0507009873554805;
  const alphaConst = 1.6732632423543772;
  const zero = ctx.full(0, x.type);
  const lambdaVal = ctx.full(lambda, x.type);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const alphaVal = ctx.full(alphaConst, x.type);
  const expX = ctx.builder.exp(x).getResult(0);
  const alphaExp = ctx.builder.mul(alphaVal, expX).getResult(0);
  const innerDeriv = ctx.builder.select(mask, lambdaVal, ctx.builder.mul(lambdaVal, alphaExp).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, innerDeriv).getResult(0)];
});

registerVJPRule('hardswish', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const negThree = ctx.full(-3, x.type);
  const three = ctx.full(3, x.type);
  const zero = ctx.full(0, x.type);
  const one = ctx.full(1, x.type);
  const two = ctx.full(2, x.type);
  const six = ctx.full(6, x.type);
  const maskLow = ctx.builder.compare(x, negThree, 'le').getResult(0);
  const maskHigh = ctx.builder.compare(x, three, 'ge').getResult(0);
  const twoXPlus3 = ctx.builder.add(ctx.builder.mul(two, x).getResult(0), three).getResult(0);
  const midDeriv = ctx.builder.div(twoXPlus3, six).getResult(0);
  const deriv = ctx.builder.select(maskLow, zero, ctx.builder.select(maskHigh, one, midDeriv).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('hardsigmoid', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const negThree = ctx.full(-3, x.type);
  const three = ctx.full(3, x.type);
  const zero = ctx.full(0, x.type);
  const sixth = ctx.full(1 / 6, x.type);
  const maskLow = ctx.builder.compare(x, negThree, 'le').getResult(0);
  const maskHigh = ctx.builder.compare(x, three, 'ge').getResult(0);
  const deriv = ctx.builder.select(maskLow, zero, ctx.builder.select(maskHigh, zero, sixth).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('embedding', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [weight, indices] = ctx.operands;
  const idxRank = indices.type.rank;

  const zeroWeight = ctx.full(0, weight.type);

  const gradWeight = ctx.builder.scatter(zeroWeight, indices, grad, {
    updateWindowDims: [idxRank],
    insertedWindowDims: [0],
    scatterDimsToOperandDims: [0],
    indexVectorDim: idxRank,
  }).getResult(0);

  return [gradWeight, null];
});
