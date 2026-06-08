import { registerVJPRule } from '../vjp_registry.js';

registerVJPRule('add', (ctx) => {
  const grad = ctx.gradOutputs[0];
  return [grad, grad];
});

registerVJPRule('sub', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const negGrad = ctx.builder.neg(grad).getResult(0);
  return [grad, negGrad];
});

registerVJPRule('mul', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [lhs, rhs] = ctx.operands;
  const gradLhs = ctx.builder.mul(grad, rhs).getResult(0);
  const gradRhs = ctx.builder.mul(grad, lhs).getResult(0);
  return [gradLhs, gradRhs];
});

registerVJPRule('div', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [lhs, rhs] = ctx.operands;
  const gradLhs = ctx.builder.div(grad, rhs).getResult(0);
  const rhsSq = ctx.builder.mul(rhs, rhs).getResult(0);
  const negLhs = ctx.builder.neg(lhs).getResult(0);
  const numerator = ctx.builder.mul(grad, negLhs).getResult(0);
  const gradRhs = ctx.builder.div(numerator, rhsSq).getResult(0);
  return [gradLhs, gradRhs];
});

registerVJPRule('neg', (ctx) => {
  const grad = ctx.gradOutputs[0];
  return [ctx.builder.neg(grad).getResult(0)];
});

registerVJPRule('pow', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [base, exp] = ctx.operands;
  const one = ctx.builder.scalarConstant(1, base.type.dtype).getResult(0);
  const oneBroadcast = ctx.builder.broadcast(one, base.type.shape, []).getResult(0);
  const expMinus1 = ctx.builder.sub(exp, oneBroadcast).getResult(0);
  const basePowExpM1 = ctx.builder.pow(base, expMinus1).getResult(0);
  const scaled = ctx.builder.mul(exp, basePowExpM1).getResult(0);
  const gradBase = ctx.builder.mul(grad, scaled).getResult(0);
  const basePowExp = ctx.builder.pow(base, exp).getResult(0);
  const logBase = ctx.builder.log(base).getResult(0);
  const product = ctx.builder.mul(basePowExp, logBase).getResult(0);
  const gradExp = ctx.builder.mul(grad, product).getResult(0);
  return [gradBase, gradExp];
});
