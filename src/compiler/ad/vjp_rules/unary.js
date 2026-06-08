import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('exp', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  return [ctx.builder.mul(grad, result).getResult(0)];
});

registerVJPRule('log', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  return [ctx.builder.div(grad, x).getResult(0)];
});

registerVJPRule('sqrt', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const two = ctx.builder.scalarConstant(2, result.type.dtype).getResult(0);
  const twoBroadcast = ctx.builder.broadcast(two, result.type.shape, []).getResult(0);
  const twoSqrt = ctx.builder.mul(twoBroadcast, result).getResult(0);
  return [ctx.builder.div(grad, twoSqrt).getResult(0)];
});

registerVJPRule('tanh', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const tanhSq = ctx.builder.mul(result, result).getResult(0);
  const one = ctx.builder.scalarConstant(1, result.type.dtype).getResult(0);
  const oneBroadcast = ctx.builder.broadcast(one, result.type.shape, []).getResult(0);
  const oneMinusTanhSq = ctx.builder.sub(oneBroadcast, tanhSq).getResult(0);
  return [ctx.builder.mul(grad, oneMinusTanhSq).getResult(0)];
});

registerVJPRule('sigmoid', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const one = ctx.builder.scalarConstant(1, result.type.dtype).getResult(0);
  const oneBroadcast = ctx.builder.broadcast(one, result.type.shape, []).getResult(0);
  const oneMinusSig = ctx.builder.sub(oneBroadcast, result).getResult(0);
  const sigGrad = ctx.builder.mul(result, oneMinusSig).getResult(0);
  return [ctx.builder.mul(grad, sigGrad).getResult(0)];
});

registerVJPRule('relu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const zero = ctx.builder.scalarConstant(0, x.type.dtype).getResult(0);
  const zeroBroadcast = ctx.builder.broadcast(zero, x.type.shape, []).getResult(0);
  const mask = ctx.builder.compare(x, zeroBroadcast, 'gt').getResult(0);
  return [ctx.builder.select(mask, grad, zeroBroadcast).getResult(0)];
});

registerVJPRule('gelu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const sqrt2OverPi = ctx.builder.scalarConstant(0.7978845608, dtype).getResult(0);
  const sqrt2OverPiBr = ctx.builder.broadcast(sqrt2OverPi, shape, []).getResult(0);
  const coeff = ctx.builder.scalarConstant(0.044715, dtype).getResult(0);
  const coeffBr = ctx.builder.broadcast(coeff, shape, []).getResult(0);
  const half = ctx.builder.scalarConstant(0.5, dtype).getResult(0);
  const halfBr = ctx.builder.broadcast(half, shape, []).getResult(0);
  const one = ctx.builder.scalarConstant(1, dtype).getResult(0);
  const oneBr = ctx.builder.broadcast(one, shape, []).getResult(0);
  const three = ctx.builder.scalarConstant(3, dtype).getResult(0);
  const threeBr = ctx.builder.broadcast(three, shape, []).getResult(0);
  const xCubed = ctx.builder.mul(x, ctx.builder.mul(x, x).getResult(0)).getResult(0);
  const inner = ctx.builder.add(x, ctx.builder.mul(coeffBr, xCubed).getResult(0)).getResult(0);
  const scaled = ctx.builder.mul(sqrt2OverPiBr, inner).getResult(0);
  const tanhVal = ctx.builder.tanh(scaled).getResult(0);
  const tanhSq = ctx.builder.mul(tanhVal, tanhVal).getResult(0);
  const sech2 = ctx.builder.sub(oneBr, tanhSq).getResult(0);
  const xSq = ctx.builder.mul(x, x).getResult(0);
  const threeCoeffXSq = ctx.builder.mul(threeBr, ctx.builder.mul(coeffBr, xSq).getResult(0)).getResult(0);
  const innerDeriv = ctx.builder.add(oneBr, threeCoeffXSq).getResult(0);
  const derivScaled = ctx.builder.mul(sqrt2OverPiBr, innerDeriv).getResult(0);
  const secondTerm = ctx.builder.mul(halfBr, ctx.builder.mul(derivScaled, sech2).getResult(0)).getResult(0);
  const firstTerm = ctx.builder.mul(halfBr, ctx.builder.add(oneBr, tanhVal).getResult(0)).getResult(0);
  const totalDeriv = ctx.builder.add(firstTerm, secondTerm).getResult(0);
  return [ctx.builder.mul(grad, totalDeriv).getResult(0)];
});

registerVJPRule('silu', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const sig = ctx.builder.sigmoid(x).getResult(0);
  const one = ctx.builder.scalarConstant(1, dtype).getResult(0);
  const oneBr = ctx.builder.broadcast(one, shape, []).getResult(0);
  const oneMinusSig = ctx.builder.sub(oneBr, sig).getResult(0);
  const xTimesOneMinusSig = ctx.builder.mul(x, oneMinusSig).getResult(0);
  const deriv = ctx.builder.add(sig, ctx.builder.mul(sig, xTimesOneMinusSig).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('sin', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const cosX = ctx.builder.cos(x).getResult(0);
  return [ctx.builder.mul(grad, cosX).getResult(0)];
});

registerVJPRule('cos', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const sinX = ctx.builder.sin(x).getResult(0);
  const negSin = ctx.builder.neg(sinX).getResult(0);
  return [ctx.builder.mul(grad, negSin).getResult(0)];
});

registerVJPRule('abs', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const signX = ctx.builder.sign(x).getResult(0);
  return [ctx.builder.mul(grad, signX).getResult(0)];
});

registerVJPRule('erf', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const coeff = ctx.builder.broadcast(ctx.builder.scalarConstant(2.0 / Math.sqrt(Math.PI), dtype).getResult(0), shape, []).getResult(0);
  const xSq = ctx.builder.mul(x, x).getResult(0);
  const negXSq = ctx.builder.neg(xSq).getResult(0);
  const expNegXSq = ctx.builder.exp(negXSq).getResult(0);
  const deriv = ctx.builder.mul(coeff, expNegXSq).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('log2', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const ln2 = ctx.builder.broadcast(ctx.builder.scalarConstant(Math.LN2, dtype).getResult(0), shape, []).getResult(0);
  const xLn2 = ctx.builder.mul(x, ln2).getResult(0);
  return [ctx.builder.div(grad, xLn2).getResult(0)];
});

registerVJPRule('log10', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const ln10 = ctx.builder.broadcast(ctx.builder.scalarConstant(Math.LN10, dtype).getResult(0), shape, []).getResult(0);
  const xLn10 = ctx.builder.mul(x, ln10).getResult(0);
  return [ctx.builder.div(grad, xLn10).getResult(0)];
});

registerVJPRule('exp2', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const dtype = result.type.dtype;
  const shape = result.type.shape;
  const ln2 = ctx.builder.broadcast(ctx.builder.scalarConstant(Math.LN2, dtype).getResult(0), shape, []).getResult(0);
  const deriv = ctx.builder.mul(result, ln2).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('square', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [x] = ctx.operands;
  const dtype = x.type.dtype;
  const shape = x.type.shape;
  const two = ctx.builder.broadcast(ctx.builder.scalarConstant(2, dtype).getResult(0), shape, []).getResult(0);
  const deriv = ctx.builder.mul(two, x).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('reciprocal', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const result = ctx.results[0];
  const resultSq = ctx.builder.mul(result, result).getResult(0);
  const negResultSq = ctx.builder.neg(resultSq).getResult(0);
  return [ctx.builder.mul(grad, negResultSq).getResult(0)];
});
