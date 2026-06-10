import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('softmax', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const axis = ctx.op.getAttr('axis');
  const dtype = result.type.dtype;
  const shape = result.type.shape;

  const gradTimesSoftmax = ctx.builder.mul(grad, result).getResult(0);
  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const sumGradSoftmax = ctx.builder.reduce(gradTimesSoftmax, initConst, [axis], 'sum').getResult(0);
  const broadcastDims = [];
  for (let i = 0; i < shape.length; i++) {
    if (i !== axis) broadcastDims.push(i);
  }
  const sumBroadcast = ctx.builder.broadcast(sumGradSoftmax, shape, broadcastDims).getResult(0);
  const shifted = ctx.builder.sub(grad, sumBroadcast).getResult(0);
  return [ctx.builder.mul(result, shifted).getResult(0)];
});

registerVJPRule('log_softmax', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const [input] = ctx.operands;
  const axis = ctx.op.getAttr('axis');
  const dtype = result.type.dtype;
  const shape = result.type.shape;

  const softmaxVal = ctx.builder.exp(result).getResult(0);
  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const sumGrad = ctx.builder.reduce(grad, initConst, [axis], 'sum').getResult(0);
  const broadcastDims = [];
  for (let i = 0; i < shape.length; i++) {
    if (i !== axis) broadcastDims.push(i);
  }
  const sumBroadcast = ctx.builder.broadcast(sumGrad, shape, broadcastDims).getResult(0);
  const softmaxTimesSum = ctx.builder.mul(softmaxVal, sumBroadcast).getResult(0);
  return [ctx.builder.sub(grad, softmaxTimesSum).getResult(0)];
});

registerVJPRule('layer_norm', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [input, gamma] = ctx.operands;
  const axis = ctx.op.getAttr('axis');
  const epsilon = ctx.op.getAttr('epsilon');
  const dtype = input.type.dtype;
  const shape = input.type.shape;
  const normDims = [axis < 0 ? shape.length + axis : axis];
  const normSize = shape[normDims[0]];

  const batchDims = [];
  for (let i = 0; i < shape.length; i++) {
    if (!normDims.includes(i)) batchDims.push(i);
  }

  const initConst = ctx.builder.scalarConstant(0, dtype).getResult(0);
  const mean = ctx.builder.reduce(input, initConst, normDims, 'mean').getResult(0);
  const meanBr = ctx.builder.broadcast(mean, shape, batchDims).getResult(0);
  const centered = ctx.builder.sub(input, meanBr).getResult(0);
  const centeredSq = ctx.builder.mul(centered, centered).getResult(0);
  const variance = ctx.builder.reduce(centeredSq, initConst, normDims, 'mean').getResult(0);
  const epsConst = ctx.builder.scalarConstant(epsilon, dtype).getResult(0);
  const epsBr = ctx.builder.broadcast(epsConst, variance.type.shape, []).getResult(0);
  const varPlusEps = ctx.builder.add(variance, epsBr).getResult(0);
  const rstd = ctx.builder.rsqrt(varPlusEps).getResult(0);
  const rstdBr = ctx.builder.broadcast(rstd, shape, batchDims).getResult(0);

  const gammaBrDims = [];
  for (let i = 0; i < shape.length; i++) {
    if (normDims.includes(i)) gammaBrDims.push(i);
  }
  const gammaBr = ctx.builder.broadcast(gamma, shape, gammaBrDims).getResult(0);

  const gradTimesGamma = ctx.builder.mul(grad, gammaBr).getResult(0);
  const nConst = ctx.builder.scalarConstant(normSize, dtype).getResult(0);
  const nBr = ctx.builder.broadcast(nConst, shape, []).getResult(0);

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

registerVJPRule('elu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const result = ctx.results[0];
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const alpha = ctx.op.getAttr('alpha') ?? 1.0;
  const zero = ctx.builder.broadcast(ctx.builder.scalarConstant(0, dtype).getResult(0), shape, []).getResult(0);
  const one = ctx.builder.broadcast(ctx.builder.scalarConstant(1, dtype).getResult(0), shape, []).getResult(0);
  const alphaVal = ctx.builder.broadcast(ctx.builder.scalarConstant(alpha, dtype).getResult(0), shape, []).getResult(0);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const negDeriv = ctx.builder.add(result, alphaVal).getResult(0);
  const deriv = ctx.builder.select(mask, one, negDeriv).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('leaky_relu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const slope = ctx.op.getAttr('negative_slope') ?? 0.01;
  const zero = ctx.builder.broadcast(ctx.builder.scalarConstant(0, dtype).getResult(0), shape, []).getResult(0);
  const one = ctx.builder.broadcast(ctx.builder.scalarConstant(1, dtype).getResult(0), shape, []).getResult(0);
  const slopeVal = ctx.builder.broadcast(ctx.builder.scalarConstant(slope, dtype).getResult(0), shape, []).getResult(0);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const deriv = ctx.builder.select(mask, one, slopeVal).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('celu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const alpha = ctx.op.getAttr('alpha') ?? 1.0;
  const zero = ctx.builder.broadcast(ctx.builder.scalarConstant(0, dtype).getResult(0), shape, []).getResult(0);
  const one = ctx.builder.broadcast(ctx.builder.scalarConstant(1, dtype).getResult(0), shape, []).getResult(0);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const alphaVal = ctx.builder.broadcast(ctx.builder.scalarConstant(alpha, dtype).getResult(0), shape, []).getResult(0);
  const xOverAlpha = ctx.builder.div(x, alphaVal).getResult(0);
  const negDeriv = ctx.builder.exp(xOverAlpha).getResult(0);
  const deriv = ctx.builder.select(mask, one, negDeriv).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('selu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const lambda = 1.0507009873554805;
  const alphaConst = 1.6732632423543772;
  const zero = ctx.builder.broadcast(ctx.builder.scalarConstant(0, dtype).getResult(0), shape, []).getResult(0);
  const lambdaVal = ctx.builder.broadcast(ctx.builder.scalarConstant(lambda, dtype).getResult(0), shape, []).getResult(0);
  const mask = ctx.builder.compare(x, zero, 'gt').getResult(0);
  const alphaVal = ctx.builder.broadcast(ctx.builder.scalarConstant(alphaConst, dtype).getResult(0), shape, []).getResult(0);
  const expX = ctx.builder.exp(x).getResult(0);
  const alphaExp = ctx.builder.mul(alphaVal, expX).getResult(0);
  const innerDeriv = ctx.builder.select(mask, lambdaVal, ctx.builder.mul(lambdaVal, alphaExp).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, innerDeriv).getResult(0)];
});

registerVJPRule('hardswish', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const negThree = ctx.builder.broadcast(ctx.builder.scalarConstant(-3, dtype).getResult(0), shape, []).getResult(0);
  const three = ctx.builder.broadcast(ctx.builder.scalarConstant(3, dtype).getResult(0), shape, []).getResult(0);
  const zero = ctx.builder.broadcast(ctx.builder.scalarConstant(0, dtype).getResult(0), shape, []).getResult(0);
  const one = ctx.builder.broadcast(ctx.builder.scalarConstant(1, dtype).getResult(0), shape, []).getResult(0);
  const two = ctx.builder.broadcast(ctx.builder.scalarConstant(2, dtype).getResult(0), shape, []).getResult(0);
  const six = ctx.builder.broadcast(ctx.builder.scalarConstant(6, dtype).getResult(0), shape, []).getResult(0);
  const maskLow = ctx.builder.compare(x, negThree, 'le').getResult(0);
  const maskHigh = ctx.builder.compare(x, three, 'ge').getResult(0);
  const twoXPlus3 = ctx.builder.add(ctx.builder.mul(two, x).getResult(0), three).getResult(0);
  const midDeriv = ctx.builder.div(twoXPlus3, six).getResult(0);
  const deriv = ctx.builder.select(maskLow, zero, ctx.builder.select(maskHigh, one, midDeriv).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('hardsigmoid', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const negThree = ctx.builder.broadcast(ctx.builder.scalarConstant(-3, dtype).getResult(0), shape, []).getResult(0);
  const three = ctx.builder.broadcast(ctx.builder.scalarConstant(3, dtype).getResult(0), shape, []).getResult(0);
  const zero = ctx.builder.broadcast(ctx.builder.scalarConstant(0, dtype).getResult(0), shape, []).getResult(0);
  const sixth = ctx.builder.broadcast(ctx.builder.scalarConstant(1 / 6, dtype).getResult(0), shape, []).getResult(0);
  const maskLow = ctx.builder.compare(x, negThree, 'le').getResult(0);
  const maskHigh = ctx.builder.compare(x, three, 'ge').getResult(0);
  const deriv = ctx.builder.select(maskLow, zero, ctx.builder.select(maskHigh, zero, sixth).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('embedding', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [weight, indices] = ctx.operands;
  const weightShape = weight.type.shape;
  const weightDtype = weight.type.dtype;
  const idxRank = indices.type.rank;
  const idxShape = indices.type.shape;

  const zeroScalar = ctx.builder.scalarConstant(0, weightDtype).getResult(0);
  const zeroWeight = ctx.builder.broadcast(zeroScalar, weightShape, []).getResult(0);

  const gradWeight = ctx.builder.scatter(zeroWeight, indices, grad, {
    updateWindowDims: [idxRank],
    insertedWindowDims: [0],
    scatterDimsToOperandDims: [0],
    indexVectorDim: idxRank,
  }).getResult(0);

  return [gradWeight, null];
});
