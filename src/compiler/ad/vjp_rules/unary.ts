import { registerVJPRule } from '../vjp_registry.js';
import type { TensorValue, VJPContext } from '../vjp_registry.js';
import { DIGAMMA_SHIFT, DIGAMMA_SERIES } from '../../../util/special_math.js';

registerVJPRule('exp', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  return [ctx.builder.mul(grad, result).getResult(0)];
});

registerVJPRule('log', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  return [ctx.builder.div(grad, x).getResult(0)];
});

registerVJPRule('sqrt', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const twoBroadcast = ctx.full(2, result.type);
  const twoSqrt = ctx.builder.mul(twoBroadcast, result).getResult(0);
  return [ctx.builder.div(grad, twoSqrt).getResult(0)];
});

registerVJPRule('tanh', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const tanhSq = ctx.builder.mul(result, result).getResult(0);
  const oneMinusTanhSq = ctx.builder.sub(ctx.full(1, result.type), tanhSq).getResult(0);
  return [ctx.builder.mul(grad, oneMinusTanhSq).getResult(0)];
});

registerVJPRule('sigmoid', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const oneMinusSig = ctx.builder.sub(ctx.full(1, result.type), result).getResult(0);
  const sigGrad = ctx.builder.mul(result, oneMinusSig).getResult(0);
  return [ctx.builder.mul(grad, sigGrad).getResult(0)];
});

registerVJPRule('relu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const zeroBroadcast = ctx.full(0, x.type);
  const mask = ctx.builder.compare(x, zeroBroadcast, 'gt').getResult(0);
  return [ctx.builder.select(mask, grad, zeroBroadcast).getResult(0)];
});

registerVJPRule('gelu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const c = ctx.full(1.702, x.type);
  const one = ctx.full(1, x.type);
  const cx = ctx.builder.mul(c, x).getResult(0);
  const s = ctx.builder.sigmoid(cx).getResult(0);
  const oneMinusS = ctx.builder.sub(one, s).getResult(0);
  const cxOneMinusS = ctx.builder.mul(cx, oneMinusS).getResult(0);
  const factor = ctx.builder.add(one, cxOneMinusS).getResult(0);
  const deriv = ctx.builder.mul(s, factor).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('silu', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const sig = ctx.builder.sigmoid(x).getResult(0);
  const oneMinusSig = ctx.builder.sub(ctx.full(1, x.type), sig).getResult(0);
  const xTimesOneMinusSig = ctx.builder.mul(x, oneMinusSig).getResult(0);
  const deriv = ctx.builder.add(sig, ctx.builder.mul(sig, xTimesOneMinusSig).getResult(0)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('sin', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const cosX = ctx.builder.cos(x).getResult(0);
  return [ctx.builder.mul(grad, cosX).getResult(0)];
});

registerVJPRule('cos', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const sinX = ctx.builder.sin(x).getResult(0);
  const negSin = ctx.builder.neg(sinX).getResult(0);
  return [ctx.builder.mul(grad, negSin).getResult(0)];
});

registerVJPRule('abs', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const signX = ctx.builder.sign(x).getResult(0);
  return [ctx.builder.mul(grad, signX).getResult(0)];
});

function _erfDerivIR(ctx: VJPContext, x: TensorValue) {
  const coeff = ctx.full(2.0 / Math.sqrt(Math.PI), x.type);
  const xSq = ctx.builder.mul(x, x).getResult(0);
  const negXSq = ctx.builder.neg(xSq).getResult(0);
  const expNegXSq = ctx.builder.exp(negXSq).getResult(0);
  return ctx.builder.mul(coeff, expNegXSq).getResult(0);
}

function _digammaIR(ctx: VJPContext, x: TensorValue) {
  const b = ctx.builder;
  const one = ctx.full(1, x.type);
  const z = b.add(x, ctx.full(DIGAMMA_SHIFT, x.type)).getResult(0);
  const invZ = b.div(one, z).getResult(0);
  const logZ = b.log(z).getResult(0);
  let acc = b.sub(logZ, b.mul(ctx.full(0.5, x.type), invZ).getResult(0)).getResult(0);
  const inv2 = b.mul(invZ, invZ).getResult(0);
  let p = inv2;
  for (const c of DIGAMMA_SERIES) {
    acc = b.add(acc, b.mul(ctx.full(c, x.type), p).getResult(0)).getResult(0);
    p = b.mul(p, inv2).getResult(0);
  }
  for (let j = 0; j < DIGAMMA_SHIFT; j++) {
    const xj = b.add(x, ctx.full(j, x.type)).getResult(0);
    acc = b.sub(acc, b.div(one, xj).getResult(0)).getResult(0);
  }
  return acc;
}

registerVJPRule('erf', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  return [ctx.builder.mul(grad, _erfDerivIR(ctx, x)).getResult(0)];
});

registerVJPRule('erfc', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const deriv = ctx.builder.neg(_erfDerivIR(ctx, x)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('lgamma', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  return [ctx.builder.mul(grad, _digammaIR(ctx, x)).getResult(0)];
});

registerVJPRule('gamma', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const result = ctx.results[0];
  const deriv = ctx.builder.mul(result, _digammaIR(ctx, x)).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('log2', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const ln2 = ctx.full(Math.LN2, x.type);
  const xLn2 = ctx.builder.mul(x, ln2).getResult(0);
  return [ctx.builder.div(grad, xLn2).getResult(0)];
});

registerVJPRule('log10', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const ln10 = ctx.full(Math.LN10, x.type);
  const xLn10 = ctx.builder.mul(x, ln10).getResult(0);
  return [ctx.builder.div(grad, xLn10).getResult(0)];
});

registerVJPRule('exp2', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const ln2 = ctx.full(Math.LN2, result.type);
  const deriv = ctx.builder.mul(result, ln2).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('square', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const two = ctx.full(2, x.type);
  const deriv = ctx.builder.mul(two, x).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

registerVJPRule('reciprocal', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const resultSq = ctx.builder.mul(result, result).getResult(0);
  const negResultSq = ctx.builder.neg(resultSq).getResult(0);
  return [ctx.builder.mul(grad, negResultSq).getResult(0)];
});

registerVJPRule('rsqrt', (ctx) => {
  const grad = ctx.gradOutputs[0]!;
  const result = ctx.results[0];
  const resultCubed = ctx.builder.mul(ctx.builder.mul(result, result).getResult(0), result).getResult(0);
  const half = ctx.full(-0.5, result.type);
  const deriv = ctx.builder.mul(half, resultCubed).getResult(0);
  return [ctx.builder.mul(grad, deriv).getResult(0)];
});

for (const op of ['floor', 'ceil', 'round', 'sign']) {
  registerVJPRule(op, (ctx) => [ctx.full(0, ctx.operands[0].type)]);
}

registerVJPRule('mish', (ctx) => {
  const b = ctx.builder;
  const grad = ctx.gradOutputs[0]!;
  const [x] = ctx.operands;
  const one = ctx.full(1, x.type);
  const softplus = b.log(b.add(one, b.exp(x).getResult(0)).getResult(0)).getResult(0);
  const tanhSp = b.tanh(softplus).getResult(0);
  const sech2 = b.sub(one, b.mul(tanhSp, tanhSp).getResult(0)).getResult(0);
  const sigmoidX = b.sigmoid(x).getResult(0);
  const deriv = b.add(tanhSp, b.mul(x, b.mul(sech2, sigmoidX).getResult(0)).getResult(0)).getResult(0);
  return [b.mul(grad, deriv).getResult(0)];
});
