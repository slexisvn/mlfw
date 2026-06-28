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

function _minMaxVJP(ctx, cmp) {
  const grad = ctx.gradOutputs[0];
  const [a, b] = ctx.operands;
  const zeroBr = ctx.full(0, a.type);
  const mask = ctx.builder.compare(a, b, cmp).getResult(0);
  const gradA = ctx.builder.select(mask, grad, zeroBr).getResult(0);
  const gradB = ctx.builder.select(mask, zeroBr, grad).getResult(0);
  return [gradA, gradB];
}

registerVJPRule('maximum', (ctx) => _minMaxVJP(ctx, 'ge'));
registerVJPRule('minimum', (ctx) => _minMaxVJP(ctx, 'le'));

registerVJPRule('clamp', (ctx) => {
  const grad = ctx.gradOutputs[0];
  const [lo, x, hi] = ctx.operands;
  const zeroBr = ctx.full(0, x.type);
  const geLo = ctx.builder.compare(x, lo, 'ge').getResult(0);
  const gradAboveLo = ctx.builder.where(geLo, grad, zeroBr).getResult(0);
  const leHi = ctx.builder.compare(x, hi, 'le').getResult(0);
  const gradX = ctx.builder.where(leHi, gradAboveLo, zeroBr).getResult(0);
  return [null, gradX, null];
});

function _whereVJP(ctx) {
  const grad = ctx.gradOutputs[0];
  const [cond] = ctx.operands;
  const zeroBr = ctx.full(0, grad.type);
  const gradA = ctx.builder.where(cond, grad, zeroBr).getResult(0);
  const gradB = ctx.builder.where(cond, zeroBr, grad).getResult(0);
  return [null, gradA, gradB];
}

registerVJPRule('where', _whereVJP);
registerVJPRule('select', _whereVJP);

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
  const oneBroadcast = ctx.full(1, base.type);
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
